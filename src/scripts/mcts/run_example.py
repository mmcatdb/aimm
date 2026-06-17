from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random

from scripts.mcts.conditions import (
    assignment_conditions_allow,
    load_assignment_conditions,
    print_assignment_conditions,
)
from search.mcts import AssignmentConditions, DatabaseInstance, MCTSOptimizer, WorkloadQuery


@dataclass(frozen=True)
class LatencyDistribution:
    median_ms: float
    sigma: float


@dataclass(frozen=True)
class ExampleCostBreakdown:
    total_cost: float
    latency_cost: float
    storage_cost: float


class RandomLatencyModel:
    """Deterministic random latency samples backed by pair-specific distributions."""

    def __init__(
        self,
        queries: list[WorkloadQuery],
        databases: list[DatabaseInstance],
        seed: int,
    ):
        self._rng = random.Random(seed)
        self._distributions = self._build_distributions(queries, databases)
        self._sample_cache: dict[tuple[str, str], float] = {}

    def estimate_latency(self, query: WorkloadQuery, database: DatabaseInstance) -> float:
        key = (query.id, database.id)
        cached = self._sample_cache.get(key)
        if cached is not None:
            return cached

        distribution = self._distributions[key]
        sampled_latency = self._rng.lognormvariate(
            math.log(distribution.median_ms),
            distribution.sigma,
        )
        self._sample_cache[key] = sampled_latency
        return sampled_latency

    def distribution_for(self, query_id: str, database_id: str) -> LatencyDistribution:
        return self._distributions[(query_id, database_id)]

    def _build_distributions(
        self,
        queries: list[WorkloadQuery],
        databases: list[DatabaseInstance],
    ) -> dict[tuple[str, str], LatencyDistribution]:
        distributions = {}
        for query_index, query in enumerate(queries):
            preferred_index = query_index % len(databases)
            secondary_index = (query_index + 1) % len(databases)

            for database_index, database in enumerate(databases):
                if database_index == preferred_index:
                    median_ms = self._rng.uniform(8.0, 22.0)
                    sigma = self._rng.uniform(0.08, 0.18)
                elif database_index == secondary_index:
                    median_ms = self._rng.uniform(28.0, 55.0)
                    sigma = self._rng.uniform(0.12, 0.25)
                else:
                    median_ms = self._rng.uniform(70.0, 140.0)
                    sigma = self._rng.uniform(0.18, 0.35)

                distributions[(query.id, database.id)] = LatencyDistribution(median_ms, sigma)

        return distributions


class RandomStorageModel:
    """Deterministic random table/database storage costs."""

    def __init__(
        self,
        table_ids: list[str],
        databases: list[DatabaseInstance],
        seed: int,
    ):
        rng = random.Random(seed)
        self._costs = self._build_costs(table_ids, databases, rng)

    def estimate_storage_cost(self, table_id: str, database: DatabaseInstance) -> float:
        return self._costs[(table_id, database.id)]

    def storage_cost_for(self, table_id: str, database_id: str) -> float:
        return self._costs[(table_id, database_id)]

    def _build_costs(
        self,
        table_ids: list[str],
        databases: list[DatabaseInstance],
        rng: random.Random,
    ) -> dict[tuple[str, str], float]:
        costs = {}
        for table_index, table_id in enumerate(table_ids):
            base_cost = rng.uniform(20.0, 120.0) * (1.0 + table_index / len(table_ids))
            for database_index, database in enumerate(databases):
                database_multiplier = (
                    0.75 + 0.25 * database_index + rng.uniform(0.0, 0.35)
                )
                costs[(table_id, database.id)] = base_cost * database_multiplier
        return costs


def main():
    parser = argparse.ArgumentParser(
        description='Run a fake 15-query storage-aware MCTS workload-routing example.',
    )
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260528)
    parser.add_argument('--queries', type=int, default=15)
    parser.add_argument('--latency-cost-weight', type=float, default=1.0)
    parser.add_argument('--storage-cost-weight', type=float, default=0.3)
    parser.add_argument(
        '--conditions-file',
        help='Path to a JSON file with optional query/database assignment conditions.',
    )
    args = parser.parse_args()

    if args.queries <= 0:
        raise ValueError('--queries must be positive')

    workload_rng = random.Random(args.seed)
    databases = [
        DatabaseInstance('postgres-primary'),
        DatabaseInstance('mongo-cluster'),
        DatabaseInstance('neo4j-graph'),
        DatabaseInstance('postgres-analytics'),
    ]
    table_ids = [
        'customer',
        'orders',
        'lineitem',
        'nation',
        'region',
        'supplier',
        'part',
        'partsupp',
    ]
    queries = [
        WorkloadQuery(
            f'q{index + 1:02d}',
            weight=workload_rng.randint(1, 8),
            storage_table_ids=choose_query_tables(workload_rng, index, table_ids),
        )
        for index in range(args.queries)
    ]
    assignment_conditions = load_assignment_conditions(
        args.conditions_file,
        queries,
        databases,
    )

    latency_model = RandomLatencyModel(queries, databases, seed=args.seed + 1)
    storage_model = RandomStorageModel(table_ids, databases, seed=args.seed + 2)
    optimizer = MCTSOptimizer(
        queries=queries,
        databases=databases,
        estimate_latency=latency_model.estimate_latency,
        estimate_storage_cost=storage_model.estimate_storage_cost,
        latency_cost_weight=args.latency_cost_weight,
        storage_cost_weight=args.storage_cost_weight,
        random_seed=args.seed + 3,
        assignment_conditions=assignment_conditions,
    )

    result = optimizer.optimize(iterations=args.iterations)
    initial_by_query = result.initial_assignment
    sampled_oracle_assignment = choose_sampled_latency_oracle_assignment(
        queries,
        databases,
        latency_model,
        assignment_conditions,
    )
    sampled_oracle_cost = assignment_cost(
        queries,
        optimizer.database_by_id,
        sampled_oracle_assignment,
        latency_model,
        storage_model,
        args.latency_cost_weight,
        args.storage_cost_weight,
    )

    print(f'Ran MCTS for {len(queries)} fake queries and {len(databases)} databases')
    print(
        f'Cost weights: latency={args.latency_cost_weight:g}, '
        f'storage={args.storage_cost_weight:g}'
    )
    print(f'Iterations completed: {result.iterations_completed}')
    print(f'Unique states visited: {result.number_of_unique_states}')
    print(
        f'Initial weighted cost: {result.initial_cost:.2f} '
        f'(latency {result.initial_latency_cost:.2f}, storage {result.initial_storage_cost:.2f})'
    )
    print(
        f'Best weighted cost:    {result.best_cost:.2f} '
        f'(latency {result.best_latency_cost:.2f}, storage {result.best_storage_cost:.2f})'
    )
    print(
        f'Greedy feasible latency assignment cost: {sampled_oracle_cost.total_cost:.2f} '
        f'(latency {sampled_oracle_cost.latency_cost:.2f}, storage {sampled_oracle_cost.storage_cost:.2f})'
    )
    print(f'Best reward:           {result.best_reward:.4f}')
    if result.initial_cost > 0:
        improvement = 100.0 * (result.initial_cost - result.best_cost) / result.initial_cost
        print(f'Improvement:           {improvement:.1f}%')
    if sampled_oracle_cost.total_cost > 0:
        gap = (
            100.0
            * (result.best_cost - sampled_oracle_cost.total_cost)
            / sampled_oracle_cost.total_cost
        )
        print(f'Gap to greedy feasible latency assignment: {gap:.1f}%')
    if not assignment_conditions.is_empty:
        print()
        print_assignment_conditions(assignment_conditions)

    print()
    print('Best assignment:')
    for query in queries:
        initial_database_id = initial_by_query[query.id]
        best_database_id = result.best_assignment[query.id]
        best_latency = latency_model.estimate_latency(
            query,
            optimizer.database_by_id[best_database_id],
        )
        initial_latency = latency_model.estimate_latency(
            query,
            optimizer.database_by_id[initial_database_id],
        )
        print(
            f'  {query.id} weight={query.weight:g}: '
            f'{best_database_id} '
            f'({best_latency:.2f} ms, initial {initial_database_id}={initial_latency:.2f} ms, '
            f'tables={", ".join(sorted(query.storage_table_ids or []))})'
        )

    print()
    print('Stored tables by database for best assignment:')
    for database_id, stored_table_ids in storage_tables_by_database(queries, result.best_assignment).items():
        print(f'  {database_id}: {", ".join(sorted(stored_table_ids))}')

    print()
    print('Latency distributions by query/database, shown as median ms +/- lognormal sigma:')
    for query in queries:
        parts = []
        for database in databases:
            distribution = latency_model.distribution_for(query.id, database.id)
            parts.append(
                f'{database.id}: {distribution.median_ms:.1f} +/- {distribution.sigma:.2f}'
            )
        print(f'  {query.id}: ' + '; '.join(parts))

    print()
    print('Storage costs by table/database:')
    for table_id in table_ids:
        parts = [
            f'{database.id}: {storage_model.storage_cost_for(table_id, database.id):.1f}'
            for database in databases
        ]
        print(f'  {table_id}: ' + '; '.join(parts))


def choose_query_tables(rng: random.Random, query_index: int, table_ids: list[str]) -> set[str]:
    table_count = rng.randint(2, min(4, len(table_ids)))
    selected = {
        table_ids[query_index % len(table_ids)],
        table_ids[(query_index + 1) % len(table_ids)],
    }
    remaining = [table_id for table_id in table_ids if table_id not in selected]
    selected.update(rng.sample(remaining, table_count - len(selected)))
    return selected


def choose_sampled_latency_oracle_assignment(
    queries: list[WorkloadQuery],
    databases: list[DatabaseInstance],
    latency_model: RandomLatencyModel,
    assignment_conditions: AssignmentConditions,
) -> dict[str, str]:
    assignment = {}
    for query in queries:
        feasible_databases = [
            database
            for database in databases
            if assignment_conditions_allow(query.id, database.id, assignment_conditions)
        ]
        best_database = min(
            feasible_databases,
            key=lambda database: latency_model.estimate_latency(query, database),
        )
        assignment[query.id] = best_database.id
    return assignment


def assignment_cost(
    queries: list[WorkloadQuery],
    database_by_id: dict[str, DatabaseInstance],
    assignment: dict[str, str],
    latency_model: RandomLatencyModel,
    storage_model: RandomStorageModel,
    latency_cost_weight: float,
    storage_cost_weight: float,
) -> ExampleCostBreakdown:
    latency_cost = 0.0
    for query in queries:
        database = database_by_id[assignment[query.id]]
        latency_cost += query.weight * latency_model.estimate_latency(query, database)

    storage_cost = 0.0
    for database_id, table_ids in storage_tables_by_database(queries, assignment).items():
        database = database_by_id[database_id]
        for table_id in table_ids:
            storage_cost += storage_model.estimate_storage_cost(table_id, database)

    return ExampleCostBreakdown(
        total_cost=latency_cost_weight * latency_cost + storage_cost_weight * storage_cost,
        latency_cost=latency_cost,
        storage_cost=storage_cost,
    )


def storage_tables_by_database(
    queries: list[WorkloadQuery],
    assignment: dict[str, str],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for query in queries:
        database_id = assignment[query.id]
        output.setdefault(database_id, set()).update(query.storage_table_ids or [])
    return output


if __name__ == '__main__':
    main()
