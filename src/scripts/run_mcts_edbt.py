from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any

from bson import json_util

from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import DriverType, MongoDriver, Neo4jDriver, PostgresDriver
from core.dynamic_provider import get_dynamic_class_instance
from core.explainers.postgres_explainer import PostgresExplainer
from core.query import (
    MongoDeleteQuery,
    MongoInsertQuery,
    MongoQuery,
    MongoUpdateQuery,
    QueryInstance,
    QueryRegistry,
    create_database_id_2,
    parse_database_id,
)
from core.utils import exit_with_exception
from dynamic.common.edbt.data_generator import EdbtDataGenerator
from latency_estimation.dataset import parse_dataset_id
from latency_estimation.mongo.flat_model import load_flat_model as load_mongo_flat_model
from latency_estimation.mongo.plan_extractor import PlanExtractor as MongoPlanExtractor
from latency_estimation.neo4j.flat_model import load_flat_model as load_neo4j_flat_model
from latency_estimation.neo4j.plan_extractor import (
    DML_RE as NEO4J_DML_RE,
    PlanExtractor as Neo4jPlanExtractor,
)
from latency_estimation.postgres.flat_model import load_flat_model as load_postgres_flat_model
from providers.path_provider import PathProvider
from search.mcts import DatabaseInstance, MCTSOptimizer, WorkloadQuery


SCHEMA = 'edbt'
MCTS_TEMPLATE_NAMES = tuple(f'mcts-{index}' for index in range(17))

DEFAULT_POSTGRES_MODEL_ID = 'postgres/edbt-2-3-flat-rf'
DEFAULT_MONGO_MODEL_ID = 'mongo/tpch-2-flat-xgb-log'
DEFAULT_NEO4J_MODEL_ID = 'neo4j/tpch-2-flat-rf'

DEFAULT_STORAGE_MULTIPLIERS = {
    DriverType.POSTGRES: 0.00035,
    DriverType.MONGO: 0.0017,
    DriverType.NEO4J: 0.00026,
}

POSTGRES_QUERY_STORAGE = {
    'mcts-0': frozenset({'order', 'customer'}),
    'mcts-1': frozenset({'order', 'customer', 'order_item', 'product'}),
    'mcts-2': frozenset({'order', 'customer', 'order_item'}),
    'mcts-3': frozenset({'order', 'order_item', 'product'}),
    'mcts-4': frozenset({'order', 'order_item', 'product', 'has_category'}),
    'mcts-5': frozenset({'order', 'customer'}),
    'mcts-6': frozenset({'order', 'customer', 'order_item', 'product'}),
    'mcts-7': frozenset({'follows'}),
    'mcts-8': frozenset({'product', 'has_category', 'has_interest'}),
    'mcts-9': frozenset({'product', 'seller', 'review'}),
    'mcts-10': frozenset({'order', 'order_item'}),
    'mcts-11': frozenset({'order'}),
    'mcts-12': frozenset({'product'}),
    'mcts-13': frozenset({'review'}),
    'mcts-14': frozenset({'customer'}),
    'mcts-15': frozenset({'seller'}),
    'mcts-16': frozenset({'order_item'}),
}

MONGO_QUERY_STORAGE = {
    'mcts-0': frozenset({'order'}),
    'mcts-1': frozenset({'order'}),
    'mcts-2': frozenset({'order'}),
    'mcts-3': frozenset({'order'}),
    'mcts-4': frozenset({'order', 'product'}),
    'mcts-5': frozenset({'order'}),
    'mcts-6': frozenset({'order'}),
    'mcts-7': frozenset({'person'}),
    'mcts-8': frozenset({'person', 'product'}),
    'mcts-9': frozenset({'product'}),
    'mcts-10': frozenset({'order'}),
    'mcts-11': frozenset({'order'}),
    'mcts-12': frozenset({'product'}),
    'mcts-13': frozenset({'review'}),
    'mcts-14': frozenset({'customer'}),
    'mcts-15': frozenset({'seller'}),
    'mcts-16': frozenset({'order'}),
}

NEO4J_QUERY_STORAGE = {
    'mcts-0': frozenset({'Person', 'Customer', 'Order', 'SNAPSHOT_OF', 'PLACED'}),
    'mcts-1': frozenset({
        'Person',
        'Customer',
        'Order',
        'Product',
        'SNAPSHOT_OF',
        'PLACED',
        'HAS_ITEM',
    }),
    'mcts-2': frozenset({
        'Person',
        'Customer',
        'Order',
        'Product',
        'SNAPSHOT_OF',
        'PLACED',
        'HAS_ITEM',
    }),
    'mcts-3': frozenset({'Seller', 'Product', 'Order', 'OFFERS', 'HAS_ITEM'}),
    'mcts-4': frozenset({'Category', 'Product', 'Order', 'HAS_CATEGORY', 'HAS_ITEM'}),
    'mcts-5': frozenset({'Person', 'Customer', 'Order', 'SNAPSHOT_OF', 'PLACED'}),
    'mcts-6': frozenset({
        'Person',
        'Customer',
        'Order',
        'Product',
        'Seller',
        'SNAPSHOT_OF',
        'PLACED',
        'HAS_ITEM',
        'OFFERS',
    }),
    'mcts-7': frozenset({'Person', 'FOLLOWS'}),
    'mcts-8': frozenset({'Person', 'Category', 'Product', 'HAS_INTEREST', 'HAS_CATEGORY'}),
    'mcts-9': frozenset({'Product', 'Seller', 'Customer', 'OFFERS', 'REVIEWED'}),
    'mcts-10': frozenset({'Product', 'Order', 'HAS_ITEM'}),
    'mcts-11': frozenset({'Order'}),
    'mcts-12': frozenset({'Product'}),
    'mcts-13': frozenset({'REVIEWED'}),
    'mcts-14': frozenset({'Customer'}),
    'mcts-15': frozenset({'Seller'}),
    'mcts-16': frozenset({'HAS_ITEM'}),
}

QUERY_STORAGE_BY_DRIVER = {
    DriverType.POSTGRES: POSTGRES_QUERY_STORAGE,
    DriverType.MONGO: MONGO_QUERY_STORAGE,
    DriverType.NEO4J: NEO4J_QUERY_STORAGE,
}


@dataclass(frozen=True)
class EdbtQueryBundle:
    semantic_id: str
    template_name: str
    instance_index: int
    title: str
    weight: float
    query_by_driver: Mapping[DriverType, QueryInstance[Any]]


class EdbtStorageCostModel:
    def __init__(
        self,
        record_counts_by_driver: Mapping[DriverType, Mapping[str, int]],
        multipliers_by_driver: Mapping[DriverType, float],
    ):
        self.record_counts_by_driver = {
            driver_type: dict(counts)
            for driver_type, counts in record_counts_by_driver.items()
        }
        self.multipliers_by_driver = dict(multipliers_by_driver)

    def estimate_storage_cost(self, table_id: str, database: DatabaseInstance) -> float:
        storage_driver_type, physical_name = parse_storage_id(table_id)
        database_driver_type, _, _ = parse_database_id(database.id)
        if storage_driver_type != database_driver_type:
            return 0.0

        counts = self.record_counts_by_driver[database_driver_type]
        if physical_name not in counts:
            raise ValueError(
                f'No record count configured for {database_driver_type.value}:{physical_name}'
            )
        return counts[physical_name] * self.multipliers_by_driver[database_driver_type]


class EdbtLatencyEstimator:
    def __init__(
        self,
        config: Config,
        model_ids_by_driver: Mapping[DriverType, str],
        scale: float,
        collect_mongo_global_stats: bool,
    ):
        self.config = config
        self.path_provider = PathProvider(config)
        self.driver_provider = DriverProvider.default(config)
        self.model_ids_by_driver = dict(model_ids_by_driver)
        self.scale = scale
        self.collect_mongo_global_stats = collect_mongo_global_stats

        self.models: dict[DriverType, Any] = {}
        self.drivers: dict[DriverType, Any] = {}
        self.plan_extractors: dict[DriverType, Any] = {}
        self.mongo_global_stats: dict | None = None
        self.prediction_cache: dict[tuple[str, str], float] = {}

    def close(self):
        for driver in self.drivers.values():
            driver.close()

    def estimate_latency(self, query: WorkloadQuery, database: DatabaseInstance) -> float:
        key = (query.id, database.id)
        cached = self.prediction_cache.get(key)
        if cached is not None:
            return cached

        driver_type, _, _ = parse_database_id(database.id)
        bundle = query.payload
        if not isinstance(bundle, EdbtQueryBundle):
            raise ValueError(f'Expected EdbtQueryBundle payload for query {query.id!r}')

        instance = bundle.query_by_driver[driver_type]
        predicted = self._predict(driver_type, instance)
        self.prediction_cache[key] = predicted
        return predicted

    def _predict(self, driver_type: DriverType, instance: QueryInstance[Any]) -> float:
        if driver_type == DriverType.POSTGRES:
            model = self._model(driver_type)
            explainer = self._plan_extractor(driver_type)
            plan = explainer.fetch_plan(instance.content, do_profile=False)
            return float(model.predict_plan(plan))

        if driver_type == DriverType.MONGO:
            model = self._model(driver_type)
            extractor = self._plan_extractor(driver_type)
            query = instance.content
            if not isinstance(query, MongoQuery):
                raise TypeError(f'Expected MongoQuery content for {instance.id!r}')
            plan = extractor.explain_query(query, _is_mongo_write_query(query), do_profile=False)
            return float(model.predict_plan(plan, self._get_mongo_global_stats()))

        if driver_type == DriverType.NEO4J:
            model = self._model(driver_type)
            extractor = self._plan_extractor(driver_type)
            query = instance.content
            if not isinstance(query, str):
                raise TypeError(f'Expected Cypher string content for {instance.id!r}')
            plan = extractor.explain_query(query, bool(NEO4J_DML_RE.search(query)), do_profile=False)
            return float(model.predict_plan(plan))

        raise ValueError(f'Unsupported driver type: {driver_type}')

    def _model(self, driver_type: DriverType):
        model = self.models.get(driver_type)
        if model is not None:
            return model

        model_id = self.model_ids_by_driver[driver_type]
        _validate_model_driver_type(model_id, driver_type)
        path = self.path_provider.flat_model(model_id)
        if driver_type == DriverType.POSTGRES:
            model = load_postgres_flat_model(path)
        elif driver_type == DriverType.MONGO:
            model = load_mongo_flat_model(path)
        elif driver_type == DriverType.NEO4J:
            model = load_neo4j_flat_model(path)
        else:
            raise ValueError(f'Unsupported driver type: {driver_type}')

        self.models[driver_type] = model
        return model

    def _driver(self, driver_type: DriverType):
        driver = self.drivers.get(driver_type)
        if driver is not None:
            return driver

        if driver_type == DriverType.POSTGRES:
            driver = self.driver_provider.get_typed(PostgresDriver, SCHEMA, self.scale)
        elif driver_type == DriverType.MONGO:
            driver = self.driver_provider.get_typed(MongoDriver, SCHEMA, self.scale)
        elif driver_type == DriverType.NEO4J:
            driver = self.driver_provider.get_typed(Neo4jDriver, SCHEMA, self.scale)
        else:
            raise ValueError(f'Unsupported driver type: {driver_type}')

        self.drivers[driver_type] = driver
        return driver

    def _plan_extractor(self, driver_type: DriverType):
        extractor = self.plan_extractors.get(driver_type)
        if extractor is not None:
            return extractor

        driver = self._driver(driver_type)
        if driver_type == DriverType.POSTGRES:
            extractor = PostgresExplainer(driver, operators=None)
        elif driver_type == DriverType.MONGO:
            extractor = MongoPlanExtractor(driver)
        elif driver_type == DriverType.NEO4J:
            extractor = Neo4jPlanExtractor(driver)
        else:
            raise ValueError(f'Unsupported driver type: {driver_type}')

        self.plan_extractors[driver_type] = extractor
        return extractor

    def _get_mongo_global_stats(self) -> dict:
        if self.mongo_global_stats is not None:
            return self.mongo_global_stats

        database_id = create_database_id_2(DriverType.MONGO, SCHEMA, self.scale)
        stats = _load_cached_mongo_global_stats(self.path_provider, database_id)
        if stats is None:
            if not self.collect_mongo_global_stats:
                raise ValueError(
                    f'No cached Mongo global stats with field distributions found for {database_id!r}. '
                    'Run with --collect-mongo-global-stats to collect and cache them.'
                )
            extractor = self._plan_extractor(DriverType.MONGO)
            stats = extractor.collect_global_stats()
            _save_cached_mongo_global_stats(self.path_provider, database_id, stats)

        self.mongo_global_stats = stats
        return stats


def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Run storage-aware MCTS on real EDBT queries and flat latency models.',
    )
    add_args(parser)
    args = parser.parse_args(raw_args)

    try:
        run(args)
    except Exception as exc:
        exit_with_exception(exc)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--instances-per-template', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--latency-cost-weight', type=float, default=1.0)
    parser.add_argument('--storage-cost-weight', type=float, default=1.0)
    parser.add_argument('--postgres-model-id', default=DEFAULT_POSTGRES_MODEL_ID)
    parser.add_argument('--mongo-model-id', default=DEFAULT_MONGO_MODEL_ID)
    parser.add_argument('--neo4j-model-id', default=DEFAULT_NEO4J_MODEL_ID)
    parser.add_argument(
        '--postgres-storage-multiplier',
        type=float,
        default=DEFAULT_STORAGE_MULTIPLIERS[DriverType.POSTGRES],
    )
    parser.add_argument(
        '--mongo-storage-multiplier',
        type=float,
        default=DEFAULT_STORAGE_MULTIPLIERS[DriverType.MONGO],
    )
    parser.add_argument(
        '--neo4j-storage-multiplier',
        type=float,
        default=DEFAULT_STORAGE_MULTIPLIERS[DriverType.NEO4J],
    )
    parser.add_argument('--collect-mongo-global-stats', action='store_true')
    parser.add_argument('--describe-only', action='store_true')


def run(args: argparse.Namespace):
    if args.scale <= 0:
        raise ValueError('--scale must be positive')
    if args.iterations < 0:
        raise ValueError('--iterations must be non-negative')
    if args.instances_per_template <= 0:
        raise ValueError('--instances-per-template must be positive')

    model_ids_by_driver = {
        DriverType.POSTGRES: args.postgres_model_id,
        DriverType.MONGO: args.mongo_model_id,
        DriverType.NEO4J: args.neo4j_model_id,
    }
    multipliers_by_driver = {
        DriverType.POSTGRES: args.postgres_storage_multiplier,
        DriverType.MONGO: args.mongo_storage_multiplier,
        DriverType.NEO4J: args.neo4j_storage_multiplier,
    }

    query_bundles = build_edbt_query_bundles(args.scale, args.instances_per_template)
    storage_model = build_storage_cost_model(args.scale, multipliers_by_driver)
    databases = build_databases(args.scale)
    queries = build_workload_queries(query_bundles)

    print_setup(
        args=args,
        model_ids_by_driver=model_ids_by_driver,
        multipliers_by_driver=multipliers_by_driver,
        query_bundles=query_bundles,
        storage_model=storage_model,
        databases=databases,
    )
    if args.describe_only:
        return

    config = Config.load()
    latency_estimator = EdbtLatencyEstimator(
        config=config,
        model_ids_by_driver=model_ids_by_driver,
        scale=args.scale,
        collect_mongo_global_stats=args.collect_mongo_global_stats,
    )

    try:
        optimizer = MCTSOptimizer(
            queries=queries,
            databases=databases,
            estimate_latency=latency_estimator.estimate_latency,
            estimate_storage_cost=storage_model.estimate_storage_cost,
            latency_cost_weight=args.latency_cost_weight,
            storage_cost_weight=args.storage_cost_weight,
            random_seed=args.seed,
        )
        result = optimizer.optimize(iterations=args.iterations)
        print_result(
            result=result,
            queries=queries,
            databases=databases,
            latency_estimator=latency_estimator,
            storage_model=storage_model,
        )
    finally:
        latency_estimator.close()


def build_edbt_query_bundles(scale: float, instances_per_template: int) -> list[EdbtQueryBundle]:
    registries = {
        driver_type: get_dynamic_class_instance(QueryRegistry, driver_type, SCHEMA)
        for driver_type in DriverType
    }

    bundles = list[EdbtQueryBundle]()
    for instance_index in range(instances_per_template):
        for template_name in MCTS_TEMPLATE_NAMES:
            query_by_driver = {}
            weight: float | None = None
            title: str | None = None

            for driver_type, registry in registries.items():
                template = registry.get_template(template_name)
                if template is None:
                    raise ValueError(f'Missing EDBT template {template_name!r} for {driver_type.value}')
                query = template.generate(scale, instance_index)
                query_by_driver[driver_type] = query
                weight = float(template.weight)
                if title is None:
                    title = _extract_title(query.label)

            bundles.append(EdbtQueryBundle(
                semantic_id=f'{template_name}:{instance_index}',
                template_name=template_name,
                instance_index=instance_index,
                title=title or template_name,
                weight=weight if weight is not None else 1.0,
                query_by_driver=query_by_driver,
            ))

    return bundles


def build_workload_queries(query_bundles: list[EdbtQueryBundle]) -> list[WorkloadQuery]:
    return [
        WorkloadQuery(
            bundle.semantic_id,
            weight=bundle.weight,
            payload=bundle,
            storage_table_ids=storage_ids_for_template(bundle.template_name),
        )
        for bundle in query_bundles
    ]


def build_databases(scale: float) -> list[DatabaseInstance]:
    return [
        DatabaseInstance(create_database_id_2(driver_type, SCHEMA, scale))
        for driver_type in DriverType
    ]


def build_storage_cost_model(
    scale: float,
    multipliers_by_driver: Mapping[DriverType, float] | None = None,
) -> EdbtStorageCostModel:
    return EdbtStorageCostModel(
        expected_record_counts_by_driver(scale),
        multipliers_by_driver or DEFAULT_STORAGE_MULTIPLIERS,
    )


def expected_record_counts_by_driver(scale: float) -> dict[DriverType, dict[str, int]]:
    counts = EdbtDataGenerator().generate_counts(scale)
    order_item = int(round(counts.order * 1.555))
    has_category = int(round(counts.product * 1.45))
    has_interest = int(round(counts.person * 4.75))

    postgres_counts = {
        'person': counts.person,
        'customer': counts.order,
        'seller': counts.seller,
        'product': counts.product,
        'category': counts.category,
        'has_category': has_category,
        'has_interest': has_interest,
        'follows': counts.follows,
        'order': counts.order,
        'order_item': order_item,
        'review': counts.review,
    }
    mongo_counts = {
        'person': counts.person,
        'customer': counts.order,
        'seller': counts.seller,
        'product': counts.product,
        'category': counts.category,
        'review': counts.review,
        'order': counts.order,
    }
    neo4j_counts = {
        'Person': counts.person,
        'Customer': counts.order,
        'Seller': counts.seller,
        'Product': counts.product,
        'Category': counts.category,
        'Order': counts.order,
        'HAS_ITEM': order_item,
        'REVIEWED': counts.review,
        'HAS_CATEGORY': has_category,
        'HAS_INTEREST': has_interest,
        'FOLLOWS': counts.follows,
        'SNAPSHOT_OF': counts.order,
        'OFFERS': counts.product,
        'PLACED': counts.order,
    }
    return {
        DriverType.POSTGRES: postgres_counts,
        DriverType.MONGO: mongo_counts,
        DriverType.NEO4J: neo4j_counts,
    }


def storage_ids_for_template(template_name: str) -> frozenset[str]:
    storage_ids = set[str]()
    for driver_type in DriverType:
        storage_ids.update(namespaced_storage_ids(
            driver_type,
            physical_storage_items_for_template(template_name, driver_type),
        ))
    return frozenset(storage_ids)


def physical_storage_items_for_template(
    template_name: str,
    driver_type: DriverType,
) -> frozenset[str]:
    return QUERY_STORAGE_BY_DRIVER[driver_type][template_name]


def namespaced_storage_ids(driver_type: DriverType, physical_names: frozenset[str]) -> frozenset[str]:
    return frozenset(f'{driver_type.value}:{physical_name}' for physical_name in physical_names)


def parse_storage_id(table_id: str) -> tuple[DriverType, str]:
    driver_name, physical_name = table_id.split(':', 1)
    return DriverType(driver_name), physical_name


def full_union_storage_costs(
    scale: float,
    multipliers_by_driver: Mapping[DriverType, float] | None = None,
) -> dict[DriverType, float]:
    storage_model = build_storage_cost_model(scale, multipliers_by_driver)
    output = {}
    for driver_type in DriverType:
        database = DatabaseInstance(create_database_id_2(driver_type, SCHEMA, scale))
        output[driver_type] = sum(
            storage_model.estimate_storage_cost(table_id, database)
            for table_id in namespaced_storage_ids(
                driver_type,
                full_union_storage_items(driver_type),
            )
        )
    return output


def full_union_storage_items(driver_type: DriverType) -> frozenset[str]:
    output = set[str]()
    for template_name in MCTS_TEMPLATE_NAMES:
        output.update(physical_storage_items_for_template(template_name, driver_type))
    return frozenset(output)


def storage_ids_by_database(
    queries: list[WorkloadQuery],
    assignment: Mapping[str, str],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for query in queries:
        database_id = assignment[query.id]
        driver_type, _, _ = parse_database_id(database_id)
        for table_id in query.storage_table_ids or []:
            storage_driver_type, _ = parse_storage_id(table_id)
            if storage_driver_type == driver_type:
                output.setdefault(database_id, set()).add(table_id)
    return output


def print_setup(
    args: argparse.Namespace,
    model_ids_by_driver: Mapping[DriverType, str],
    multipliers_by_driver: Mapping[DriverType, float],
    query_bundles: list[EdbtQueryBundle],
    storage_model: EdbtStorageCostModel,
    databases: list[DatabaseInstance],
):
    print('EDBT MCTS setup')
    print(f'  scale: {args.scale:g}')
    print(f'  templates: {len(MCTS_TEMPLATE_NAMES)}')
    print(f'  instances per template: {args.instances_per_template}')
    print(f'  workload queries: {len(query_bundles)}')
    print(f'  iterations: {args.iterations}')
    print(
        f'  cost weights: latency={args.latency_cost_weight:g}, '
        f'storage={args.storage_cost_weight:g}'
    )
    print('  model ids:')
    for driver_type, model_id in model_ids_by_driver.items():
        print(f'    {driver_type.value}: {model_id}')
    print('  storage multipliers:')
    for driver_type, multiplier in multipliers_by_driver.items():
        print(f'    {driver_type.value}: {multiplier:g}')

    print()
    print('Full-union storage baseline by database:')
    database_by_driver = {
        parse_database_id(database.id)[0]: database
        for database in databases
    }
    for driver_type in DriverType:
        database = database_by_driver[driver_type]
        item_ids = namespaced_storage_ids(driver_type, full_union_storage_items(driver_type))
        cost = sum(storage_model.estimate_storage_cost(table_id, database) for table_id in item_ids)
        items = ', '.join(sorted(table_id.split(':', 1)[1] for table_id in item_ids))
        print(f'  {database.id}: {cost:.2f} ({items})')

    print()
    print('Semantic workload:')
    for bundle in query_bundles:
        print(f'  {bundle.semantic_id} weight={bundle.weight:g}: {bundle.title}')


def print_result(
    result,
    queries: list[WorkloadQuery],
    databases: list[DatabaseInstance],
    latency_estimator: EdbtLatencyEstimator,
    storage_model: EdbtStorageCostModel,
):
    database_by_id = {database.id: database for database in databases}

    print()
    print('MCTS result')
    print(f'  iterations completed: {result.iterations_completed}')
    print(f'  unique states visited: {result.number_of_unique_states}')
    print(
        f'  initial weighted cost: {result.initial_cost:.2f} '
        f'(latency {result.initial_latency_cost:.2f}, storage {result.initial_storage_cost:.2f})'
    )
    print(
        f'  best weighted cost:    {result.best_cost:.2f} '
        f'(latency {result.best_latency_cost:.2f}, storage {result.best_storage_cost:.2f})'
    )
    print(f'  best reward:           {result.best_reward:.4f}')

    print()
    print('Best assignment:')
    for query in queries:
        database_id = result.best_assignment[query.id]
        database = database_by_id[database_id]
        driver_type, _, _ = parse_database_id(database_id)
        latency = latency_estimator.estimate_latency(query, database)
        bundle = query.payload
        storage_items = sorted(
            physical_storage_items_for_template(bundle.template_name, driver_type)
            if isinstance(bundle, EdbtQueryBundle)
            else []
        )
        print(
            f'  {query.id} weight={query.weight:g}: {database_id} '
            f'({latency:.2f} ms, storage={", ".join(storage_items)})'
        )

    print()
    print('Predicted latency by query/database:')
    for query in queries:
        parts = []
        for database in databases:
            marker = '*' if result.best_assignment[query.id] == database.id else ' '
            latency = latency_estimator.estimate_latency(query, database)
            parts.append(f'{marker}{database.id}={latency:.2f} ms')
        print(f'  {query.id}: ' + '; '.join(parts))

    print()
    print('Stored physical items by database:')
    for database_id, table_ids in sorted(storage_ids_by_database(queries, result.best_assignment).items()):
        database = database_by_id[database_id]
        cost = sum(storage_model.estimate_storage_cost(table_id, database) for table_id in table_ids)
        physical_names = [table_id.split(':', 1)[1] for table_id in sorted(table_ids)]
        print(f'  {database_id}: {cost:.2f} ({", ".join(physical_names)})')


def _validate_model_driver_type(model_id: str, expected_driver_type: DriverType):
    driver_type, _ = parse_dataset_id(model_id)
    if driver_type != expected_driver_type:
        raise ValueError(
            f'Model id {model_id!r} must belong to {expected_driver_type.value!r}'
        )


def _extract_title(label: str) -> str:
    _, separator, title = label.partition(' - ')
    return title if separator else label


def _is_mongo_write_query(query: MongoQuery) -> bool:
    return isinstance(query, (MongoUpdateQuery, MongoDeleteQuery, MongoInsertQuery))


def _load_cached_mongo_global_stats(pp: PathProvider, database_id: str) -> dict | None:
    path = pp.global_stats(database_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        stats = json_util.loads(file.read())
    if not _has_mongo_field_stats(stats):
        return None
    return stats


def _save_cached_mongo_global_stats(pp: PathProvider, database_id: str, global_stats: dict):
    path = pp.global_stats(database_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json_util.dumps(global_stats))


def _has_mongo_field_stats(global_stats: dict) -> bool:
    if not isinstance(global_stats, dict) or not global_stats:
        return False
    collection_stats = [stats for stats in global_stats.values() if isinstance(stats, dict)]
    return bool(collection_stats) and all(
        MongoPlanExtractor.FIELD_STATS_KEY in stats
        for stats in collection_stats
    )


if __name__ == '__main__':
    main()
