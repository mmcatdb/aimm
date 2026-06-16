from __future__ import annotations

import argparse

from core.config import Config
from core.query import parse_database_id
from core.utils import exit_with_exception
from scripts import run_mcts_edbt
from scripts.run_mcts_edbt import (
    EdbtLatencyEstimateMatrix,
    EdbtLatencyEstimateRecord,
    EdbtQueryBundle,
)


def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Precompute EDBT MCTS query/database latency estimates.',
    )
    add_args(parser)
    args = parser.parse_args(raw_args)

    try:
        run(Config.load(), args)
    except Exception as exc:
        exit_with_exception(exc)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--instances-per-template', type=int, default=1)
    parser.add_argument('--output', help='Output JSONL path. Defaults to the MCTS cache path.')
    parser.add_argument('--postgres-model-id', default=run_mcts_edbt.DEFAULT_POSTGRES_MODEL_ID)
    parser.add_argument('--mongo-model-id', default=run_mcts_edbt.DEFAULT_MONGO_MODEL_ID)
    parser.add_argument('--neo4j-model-id', default=run_mcts_edbt.DEFAULT_NEO4J_MODEL_ID)
    parser.add_argument('--collect-mongo-global-stats', action='store_true')


def run(config: Config, args: argparse.Namespace):
    if args.scale <= 0:
        raise ValueError('--scale must be positive')
    if args.instances_per_template <= 0:
        raise ValueError('--instances-per-template must be positive')

    model_ids_by_driver = run_mcts_edbt.model_ids_by_driver_from_args(args)
    query_bundles = run_mcts_edbt.build_edbt_query_bundles(
        args.scale,
        args.instances_per_template,
    )
    queries = run_mcts_edbt.build_workload_queries(query_bundles)
    databases = run_mcts_edbt.build_databases(args.scale)
    output = args.output or run_mcts_edbt.default_edbt_latency_estimates_path(
        config,
        args.scale,
        args.instances_per_template,
    )

    latency_estimator = run_mcts_edbt.EdbtLatencyEstimator(
        config=config,
        model_ids_by_driver=model_ids_by_driver,
        scale=args.scale,
        collect_mongo_global_stats=args.collect_mongo_global_stats,
    )

    rows: list[EdbtLatencyEstimateRecord] = []
    try:
        for query in queries:
            bundle = query.payload
            if not isinstance(bundle, EdbtQueryBundle):
                raise ValueError(f'Expected EdbtQueryBundle payload for query {query.id!r}')
            for database in databases:
                driver_type, _, _ = parse_database_id(database.id)
                latency_ms = latency_estimator.estimate_latency(query, database)
                rows.append(EdbtLatencyEstimateRecord(
                    query_id=query.id,
                    database_id=database.id,
                    latency_ms=latency_ms,
                    source_query_id=bundle.query_by_driver[driver_type].id,
                ))
    finally:
        latency_estimator.close()

    matrix = EdbtLatencyEstimateMatrix(
        schema=run_mcts_edbt.SCHEMA,
        scale=args.scale,
        instances_per_template=args.instances_per_template,
        query_ids=tuple(query.id for query in queries),
        database_ids=tuple(database.id for database in databases),
        source_metadata={
            'producer': 'scripts.precompute_mcts_edbt_latencies',
            'model_ids': {
                driver_type.value: model_id
                for driver_type, model_id in model_ids_by_driver.items()
            },
        },
        rows=tuple(rows),
    )
    run_mcts_edbt.save_edbt_latency_estimates(output, matrix)

    print('Saved EDBT MCTS latency estimates')
    print(f'  output: {output}')
    print(f'  workload queries: {len(queries)}')
    print(f'  databases: {len(databases)}')
    print(f'  estimates: {len(rows)}')


if __name__ == '__main__':
    main()
