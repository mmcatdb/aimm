from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

from core.config import Config
from core.drivers import DriverType
from core.files import open_input
from core.query import parse_database_id
from core.utils import exit_with_exception
from scripts.mcts import run_edbt
from scripts.mcts.conditions import (
    DatabaseRefResolver,
    edbt_database_ref_resolver,
    load_assignment_conditions,
    print_assignment_conditions,
)
from search.mcts import (
    AssignmentConditions,
    DatabaseInstance,
    LatencyEstimator,
    PrecomputedLatencyEstimator,
    TableId,
    WorkloadQuery,
)
from search.table_placement_mcts import (
    ACTION_SELECTION_CHOICES,
    ACTION_SELECTION_UCT,
    DatabasePlacement,
    TablePlacementInput,
    TablePlacementMCTSOptimizer,
)

INITIAL_PLACEMENT_POSTGRES_ONLY = 'postgres-only'
INITIAL_PLACEMENT_FULL_UNION = 'full-union'
INITIAL_PLACEMENT_CHOICES = (
    INITIAL_PLACEMENT_POSTGRES_ONLY,
    INITIAL_PLACEMENT_FULL_UNION,
)


def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Run storage-placement MCTS on real EDBT queries and flat latency models.',
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
    parser.add_argument(
        '--action-selection',
        choices=ACTION_SELECTION_CHOICES,
        default=ACTION_SELECTION_UCT,
        help='Choose actions using UCT (default) or uniformly at random.',
    )
    parser.add_argument('--latency-cost-weight', type=float, default=1.0)
    parser.add_argument('--storage-cost-weight', type=float, default=0.3)
    parser.add_argument(
        '--query-weights',
        help=(
            'Path to a JSON file or inline JSON object with query weight overrides. '
            'Keys may be template names (mcts-0) or semantic query ids (mcts-0:0).'
        ),
    )
    parser.add_argument('--postgres-model-id', default=run_edbt.DEFAULT_POSTGRES_MODEL_ID)
    parser.add_argument('--mongo-model-id', default=run_edbt.DEFAULT_MONGO_MODEL_ID)
    parser.add_argument('--neo4j-model-id', default=run_edbt.DEFAULT_NEO4J_MODEL_ID)
    parser.add_argument(
        '--latency-estimates',
        help='Path to a precomputed EDBT MCTS latency matrix JSONL file.',
    )
    parser.add_argument(
        '--conditions-file',
        help='Path to a JSON file with optional query/database assignment conditions.',
    )
    parser.add_argument(
        '--initial-placement',
        choices=INITIAL_PLACEMENT_CHOICES,
        default=INITIAL_PLACEMENT_POSTGRES_ONLY,
    )
    parser.add_argument(
        '--initial-placement-file',
        help='Path to a JSON file describing the initial table placement.',
    )
    parser.add_argument(
        '--postgres-storage-multiplier',
        type=float,
        default=run_edbt.DEFAULT_STORAGE_MULTIPLIERS[DriverType.POSTGRES],
    )
    parser.add_argument(
        '--mongo-storage-multiplier',
        type=float,
        default=run_edbt.DEFAULT_STORAGE_MULTIPLIERS[DriverType.MONGO],
    )
    parser.add_argument(
        '--neo4j-storage-multiplier',
        type=float,
        default=run_edbt.DEFAULT_STORAGE_MULTIPLIERS[DriverType.NEO4J],
    )
    parser.add_argument('--collect-mongo-global-stats', action='store_true')
    parser.add_argument('--describe-only', action='store_true')
    parser.add_argument(
        '--print-progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Print flushed MCTS progress and best-placement updates.',
    )
    parser.add_argument(
        '--silent',
        action='store_true',
        help='Suppress all output except for the MCTS progress.',
    )


def run(args: argparse.Namespace):
    args.action_selection = getattr(args, 'action_selection', ACTION_SELECTION_UCT)
    if args.scale <= 0:
        raise ValueError('--scale must be positive')
    if args.iterations < 0:
        raise ValueError('--iterations must be non-negative')
    if args.instances_per_template <= 0:
        raise ValueError('--instances-per-template must be positive')

    is_silent = getattr(args, 'silent', False)

    model_ids_by_driver = run_edbt.model_ids_by_driver_from_args(args)
    multipliers_by_driver = run_edbt.storage_multipliers_by_driver_from_args(args)
    query_weight_overrides = run_edbt.load_query_weight_overrides(args.query_weights)

    query_bundles = run_edbt.build_edbt_query_bundles(
        args.scale,
        args.instances_per_template,
        query_weight_overrides=query_weight_overrides,
    )
    storage_model = run_edbt.build_storage_cost_model(args.scale, multipliers_by_driver)
    databases = run_edbt.build_databases(args.scale)
    queries = run_edbt.build_workload_queries(query_bundles)
    assignment_conditions = load_assignment_conditions(
        args.conditions_file,
        queries,
        databases,
        resolve_database_ref=edbt_database_ref_resolver(databases),
    )
    initial_placement = build_initial_placement_from_args(
        args,
        queries,
        databases,
    )

    if not is_silent:
        print_setup(
            args=args,
            model_ids_by_driver=model_ids_by_driver,
            multipliers_by_driver=multipliers_by_driver,
            query_bundles=query_bundles,
            storage_model=storage_model,
            databases=databases,
            queries=queries,
            initial_placement=initial_placement,
        )

    if args.describe_only:
        return

    should_close_latency_estimator = False
    if args.latency_estimates:
        latency_matrix = run_edbt.load_edbt_latency_estimates(args.latency_estimates)
        run_edbt.validate_edbt_latency_estimates(
            latency_matrix,
            scale=args.scale,
            instances_per_template=args.instances_per_template,
            queries=queries,
            databases=databases,
            assignment_conditions=assignment_conditions,
        )
        latency_estimator = PrecomputedLatencyEstimator(latency_matrix.latency_estimates())
    else:
        config = Config.load()
        latency_estimator = run_edbt.EdbtLatencyEstimator(
            config=config,
            model_ids_by_driver=model_ids_by_driver,
            scale=args.scale,
            collect_mongo_global_stats=args.collect_mongo_global_stats,
        )
        should_close_latency_estimator = True

    optimizer = TablePlacementMCTSOptimizer(
        queries=queries,
        databases=databases,
        latency_estimator=latency_estimator,
        estimate_storage_cost=storage_model.estimate_storage_cost,
        get_query_database_storage_table_ids=edbt_query_database_storage_ids,
        can_store=edbt_can_store,
        latency_cost_weight=args.latency_cost_weight,
        storage_cost_weight=args.storage_cost_weight,
        random_seed=args.seed,
        assignment_conditions=assignment_conditions,
        action_selection=args.action_selection,
        verbose=args.print_progress,
        format_placement_schema=format_edbt_placement_schema,
    )

    try:
        result = optimizer.optimize(
            iterations=args.iterations,
            initial_placement=initial_placement,
        )
        if not is_silent:
            print_result(
                result=result,
                queries=queries,
                databases=databases,
                latency_estimator=latency_estimator,
                storage_model=storage_model,
                assignment_conditions=assignment_conditions,
            )
    finally:
        if should_close_latency_estimator:
            latency_estimator.close()


def edbt_query_database_storage_ids(
    query: WorkloadQuery,
    database: DatabaseInstance,
) -> frozenset[str]:
    database_driver_type, _, _ = parse_database_id(database.id)
    output = set[str]()
    for table_id in query.storage_table_ids or []:
        storage_driver_type, _ = run_edbt.parse_storage_id(table_id)
        if storage_driver_type == database_driver_type:
            output.add(table_id)
    return frozenset(output)


def edbt_can_store(table_id: TableId, database: DatabaseInstance) -> bool:
    storage_driver_type, _ = run_edbt.parse_storage_id(str(table_id))
    database_driver_type, _, _ = parse_database_id(database.id)
    return storage_driver_type == database_driver_type


def build_initial_placement_from_args(
    args: argparse.Namespace,
    queries: Sequence[WorkloadQuery],
    databases: Sequence[DatabaseInstance],
) -> TablePlacementInput | None:
    initial_placement_file = getattr(args, 'initial_placement_file', None)
    if initial_placement_file:
        return load_initial_placement(initial_placement_file, databases)

    initial_placement = getattr(args, 'initial_placement', INITIAL_PLACEMENT_POSTGRES_ONLY)
    return build_initial_placement(initial_placement, queries, databases)


def build_initial_placement(
    initial_placement: str,
    queries: Sequence[WorkloadQuery],
    databases: Sequence[DatabaseInstance],
) -> TablePlacementInput | None:
    if initial_placement == INITIAL_PLACEMENT_FULL_UNION:
        return None

    if initial_placement == INITIAL_PLACEMENT_POSTGRES_ONLY:
        postgres_database_id = database_id_for_driver(databases, DriverType.POSTGRES)
        placement: dict[str, set[str]] = {}
        for query in queries:
            for table_id in query.storage_table_ids or []:
                storage_driver_type, _ = run_edbt.parse_storage_id(table_id)
                if storage_driver_type == DriverType.POSTGRES:
                    placement.setdefault(table_id, set()).add(postgres_database_id)
        return placement

    raise ValueError(f'Unknown initial placement mode {initial_placement!r}')


def database_id_for_driver(
    databases: Sequence[DatabaseInstance],
    driver_type: DriverType,
) -> str:
    for database in databases:
        database_driver_type, _, _ = parse_database_id(database.id)
        if database_driver_type == driver_type:
            return database.id
    raise ValueError(f'No {driver_type.value} database is available for initial placement')


def load_initial_placement(
    path: str,
    databases: Sequence[DatabaseInstance],
) -> dict[str, set[str]]:
    try:
        with open_input(path) as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Initial placement file {path!r} is not valid JSON: {exc.msg}'
        ) from exc
    except OSError as exc:
        raise ValueError(f'Could not read initial placement file {path!r}: {exc}') from exc

    return parse_initial_placement(
        data,
        databases,
        resolve_database_ref=edbt_database_ref_resolver(databases),
    )


def parse_initial_placement(
    data: object,
    databases: Sequence[DatabaseInstance],
    resolve_database_ref: DatabaseRefResolver | None = None,
) -> dict[str, set[str]]:
    if not isinstance(data, dict):
        raise ValueError('Initial placement file must contain a JSON object')

    resolver = resolve_database_ref or edbt_database_ref_resolver(databases)
    if 'by_database' in data or 'by_table' in data:
        allowed_keys = {'by_database', 'by_table'}
        unknown_keys = sorted(set(data) - allowed_keys)
        if unknown_keys:
            raise ValueError(
                'Initial placement file contains unknown keys: '
                + ', '.join(repr(key) for key in unknown_keys)
            )
        if 'by_database' in data and 'by_table' in data:
            raise ValueError('Initial placement file must not contain both by_database and by_table')
        if 'by_database' in data:
            return parse_initial_placement_by_database(data['by_database'], resolver)
        return parse_initial_placement_by_table(data['by_table'], resolver)

    if _all_keys_are_database_refs(data, resolver):
        return parse_initial_placement_by_database(data, resolver)
    return parse_initial_placement_by_table(data, resolver)


def parse_initial_placement_by_database(
    data: object,
    resolve_database_ref: DatabaseRefResolver,
) -> dict[str, set[str]]:
    if not isinstance(data, dict):
        raise ValueError('Initial placement by_database must be an object')

    placement: dict[str, set[str]] = {}
    for database_ref, table_ids in data.items():
        if not isinstance(database_ref, str):
            raise ValueError(f'Initial placement database id must be a string: {database_ref!r}')
        database_id = resolve_database_ref(database_ref)
        for table_id in _parse_table_id_list(table_ids, f'Initial placement for {database_ref!r}'):
            placement.setdefault(table_id, set()).add(database_id)
    return placement


def parse_initial_placement_by_table(
    data: object,
    resolve_database_ref: DatabaseRefResolver,
) -> dict[str, set[str]]:
    if not isinstance(data, dict):
        raise ValueError('Initial placement by_table must be an object')

    placement: dict[str, set[str]] = {}
    for table_id, database_refs in data.items():
        if not isinstance(table_id, str):
            raise ValueError(f'Initial placement table id must be a string: {table_id!r}')
        placement[table_id] = {
            resolve_database_ref(database_ref)
            for database_ref in _parse_database_ref_list(database_refs, table_id)
        }
    return placement


def _all_keys_are_database_refs(
    data: Mapping[Any, Any],
    resolve_database_ref: DatabaseRefResolver,
) -> bool:
    if not data:
        return False

    for key in data:
        if not isinstance(key, str):
            return False
        try:
            resolve_database_ref(key)
        except ValueError:
            return False
    return True


def _parse_table_id_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{label} must be an array of storage table ids')
    output = []
    for table_id in value:
        if not isinstance(table_id, str):
            raise ValueError(f'{label} contains a non-string storage table id: {table_id!r}')
        output.append(table_id)
    return output


def _parse_database_ref_list(value: object, table_id: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(
            f'Initial placement database refs for table {table_id!r} must be a string or array'
        )
    output = []
    for database_ref in value:
        if not isinstance(database_ref, str):
            raise ValueError(
                f'Initial placement database refs for table {table_id!r} '
                f'must be strings: {database_ref!r}'
            )
        output.append(database_ref)
    return output


def format_edbt_placement_schema(placement: DatabasePlacement) -> dict[str, list[str]]:
    output = {}
    for database_id, table_ids in placement.items():
        driver_type, _, _ = parse_database_id(database_id)
        output[run_edbt.format_driver_label(driver_type)] = sorted(
            run_edbt.parse_storage_id(table_id)[1]
            for table_id in table_ids
        )
    return output


def print_setup(
    args: argparse.Namespace,
    model_ids_by_driver: Mapping[DriverType, str],
    multipliers_by_driver: Mapping[DriverType, float],
    query_bundles,
    storage_model,
    databases: list[DatabaseInstance],
    queries: list[WorkloadQuery],
    initial_placement: TablePlacementInput | None,
):
    print('EDBT table-placement MCTS setup')
    print(f'  scale: {args.scale:g}')
    print(f'  templates: {len(run_edbt.MCTS_TEMPLATE_NAMES)}')
    print(f'  instances per template: {args.instances_per_template}')
    print(f'  workload queries: {len(query_bundles)}')
    print(f'  iterations: {args.iterations}')
    print(f'  action selection: {args.action_selection}')
    print(
        f'  cost weights: latency={args.latency_cost_weight:g}, '
        f'storage={args.storage_cost_weight:g}'
    )
    if getattr(args, 'initial_placement_file', None):
        print(f'  initial placement file: {args.initial_placement_file}')
    else:
        print(f'  initial placement: {args.initial_placement}')
    latency_estimates_path = args.latency_estimates
    if latency_estimates_path:
        print(f'  latency estimates: {latency_estimates_path}')
        print('  model ids: ignored for offline MCTS')
    else:
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
        item_ids = run_edbt.namespaced_storage_ids(
            driver_type,
            run_edbt.full_union_storage_items(driver_type),
        )
        cost = sum(storage_model.estimate_storage_cost(table_id, database) for table_id in item_ids)
        items = ', '.join(sorted(table_id.split(':', 1)[1] for table_id in item_ids))
        print(f'  {database.id}: {cost:.2f} ({items})')

    initial_database_placement = initial_placement_to_database_placement(
        initial_placement,
        queries,
        databases,
    )
    print()
    print('Initial stored physical items by database:')
    print_database_placement(initial_database_placement, databases, storage_model)

    print()
    print('Semantic workload:')
    for bundle in query_bundles:
        print(f'  {bundle.semantic_id} weight={bundle.weight:g}: {bundle.title}')


def initial_placement_to_database_placement(
    initial_placement: TablePlacementInput | None,
    queries: Sequence[WorkloadQuery],
    databases: Sequence[DatabaseInstance],
) -> DatabasePlacement:
    if initial_placement is None:
        placement: DatabasePlacement = {}
        for query in queries:
            for database in databases:
                for table_id in edbt_query_database_storage_ids(query, database):
                    placement.setdefault(database.id, set()).add(table_id)
        return placement

    output: DatabasePlacement = {}
    if isinstance(initial_placement, Mapping):
        for table_id, database_ids in initial_placement.items():
            for database_id in database_ids:
                output.setdefault(database_id, set()).add(table_id)
        return output

    for table_id, database_id in initial_placement:
        output.setdefault(database_id, set()).add(table_id)
    return output


def print_result(
    result,
    queries: list[WorkloadQuery],
    databases: list[DatabaseInstance],
    latency_estimator: LatencyEstimator,
    storage_model,
    assignment_conditions: AssignmentConditions,
):
    database_by_id = {database.id: database for database in databases}

    print()
    print('Table-placement MCTS result')
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
    if result.initial_cost > 0 and math.isfinite(result.best_cost):
        improvement = 100.0 * (result.initial_cost - result.best_cost) / result.initial_cost
        print(f'  improvement:           {improvement:.1f}%')
    if not assignment_conditions.is_empty:
        print()
        print_assignment_conditions(assignment_conditions)

    print()
    print('Best query routing:')
    for query in queries:
        database_id = result.best_assignment[query.id]
        database = database_by_id[database_id]
        latency = latency_estimator.estimate_latency(query, database)
        storage_items = sorted(
            run_edbt.parse_storage_id(table_id)[1]
            for table_id in edbt_query_database_storage_ids(query, database)
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
    print('Initial stored physical items by database:')
    print_database_placement(result.initial_placement, databases, storage_model)

    print()
    print('Best stored physical items by database:')
    print_database_placement(result.best_placement, databases, storage_model)


def print_database_placement(
    placement: DatabasePlacement,
    databases: Sequence[DatabaseInstance],
    storage_model,
):
    database_by_id = {database.id: database for database in databases}
    for database_id in sorted(placement):
        table_ids = placement[database_id]
        database = database_by_id[database_id]
        cost = sum(storage_model.estimate_storage_cost(table_id, database) for table_id in table_ids)
        physical_names = [table_id.split(':', 1)[1] for table_id in sorted(table_ids)]
        print(f'  {database_id}: {cost:.2f} ({", ".join(physical_names)})')
    if not placement:
        print('  none')


if __name__ == '__main__':
    main()
