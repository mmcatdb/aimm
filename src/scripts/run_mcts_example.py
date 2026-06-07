from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random

from search.mcts import DatabaseInstance, MCTSOptimizer, WorkloadQuery


@dataclass(frozen=True)
class LatencyDistribution:
    median_ms: float
    sigma: float


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


def main():
    parser = argparse.ArgumentParser(
        description='Run a fake 15-query MCTS workload-routing example.',
    )
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260528)
    parser.add_argument('--queries', type=int, default=15)
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
    queries = [
        WorkloadQuery(f'q{index + 1:02d}', weight=workload_rng.randint(1, 8))
        for index in range(args.queries)
    ]

    latency_model = RandomLatencyModel(queries, databases, seed=args.seed + 1)
    optimizer = MCTSOptimizer(
        queries=queries,
        databases=databases,
        estimate_latency=latency_model.estimate_latency,
        random_seed=args.seed + 2,
    )

    result = optimizer.optimize(iterations=args.iterations)
    initial_by_query = result.initial_assignment
    sampled_oracle_assignment = choose_sampled_oracle_assignment(queries, databases, latency_model)
    sampled_oracle_cost = assignment_cost(queries, optimizer.database_by_id, sampled_oracle_assignment, latency_model)

    print(f'Ran MCTS for {len(queries)} fake queries and {len(databases)} databases')
    print(f'Iterations completed: {result.iterations_completed}')
    print(f'Unique states visited: {result.number_of_unique_states}')
    print(f'Initial weighted cost: {result.initial_cost:.2f} ms')
    print(f'Best weighted cost:    {result.best_cost:.2f} ms')
    print(f'Sampled oracle cost:   {sampled_oracle_cost:.2f} ms')
    print(f'Best reward:           {result.best_reward:.4f}')
    if result.initial_cost > 0:
        improvement = 100.0 * (result.initial_cost - result.best_cost) / result.initial_cost
        print(f'Improvement:           {improvement:.1f}%')
    if sampled_oracle_cost > 0:
        gap = 100.0 * (result.best_cost - sampled_oracle_cost) / sampled_oracle_cost
        print(f'Gap to sampled oracle: {gap:.1f}%')

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
            f'({best_latency:.2f} ms, initial {initial_database_id}={initial_latency:.2f} ms)'
        )

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


def choose_sampled_oracle_assignment(
    queries: list[WorkloadQuery],
    databases: list[DatabaseInstance],
    latency_model: RandomLatencyModel,
) -> dict[str, str]:
    assignment = {}
    for query in queries:
        best_database = min(
            databases,
            key=lambda database: latency_model.estimate_latency(query, database),
        )
        assignment[query.id] = best_database.id
    return assignment


def assignment_cost(
    queries: list[WorkloadQuery],
    database_by_id: dict[str, DatabaseInstance],
    assignment: dict[str, str],
    latency_model: RandomLatencyModel,
) -> float:
    total = 0.0
    for query in queries:
        database = database_by_id[assignment[query.id]]
        total += query.weight * latency_model.estimate_latency(query, database)
    return total


if __name__ == '__main__':
    main()
