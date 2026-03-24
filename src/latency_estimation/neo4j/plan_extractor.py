import time
from typing_extensions import override
import numpy as np
from typing import Any
from common.drivers import Neo4jDriver, cypher
from common.utils import ProgressTracker, print_warning, time_quantity
from common.query_registry import QueryDefMap
from latency_estimation.common import ArrayDataset
from latency_estimation.feature_extractor import BaseDatasetItem
from latency_estimation.neo4j.feature_extractor import FeatureExtractor

class Neo4jItem(BaseDatasetItem):
    def __init__(self, query: str, plan: dict, execution_time: float):
        super().__init__(plan, execution_time)
        self.query = query

    @override
    def query_string(self) -> str:
        return self.query

class PlanExtractor:
    """
    Extract query execution plans and execution times from Neo4j.
    This module handles:
    - Executing queries with EXPLAIN (for plan) and timing actual execution
    - Parsing the plan structure
    - Recording ground truth latencies
    """
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    def create_dataset(self, queries: list[str], num_runs: int, def_map: QueryDefMap[str] | None = None) -> ArrayDataset[Neo4jItem]:
        """
        Collect a workload of query plans and execution times.
        Args:
            queries: List of Cypher queries to collect plans for
            num_runs: Number of executions per query for averaging
        """
        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        items: list[Neo4jItem] = []

        for i, query in enumerate(queries):
            try:
                # Get plan and execution time
                plan = self.explain_plan(query)
                execution_time, _, _ = self.measure_query(query, num_runs)
                items.append(Neo4jItem(query, plan, execution_time))
                progress.track()

            except Exception as e:
                query_def = def_map.get(id(query)) if def_map else None
                if query_def:
                    print_warning(f'\nCould not execute query {query_def.label()}.', e)
                else:
                    print_warning(f'\nCould not execute query on index {i}.', e)
                print()

        dataset = ArrayDataset(items)
        progress.finish()

        print(f'\nCollected {len(dataset)} query plans.')
        return dataset

    def explain_plan(self, query: str) -> dict:
        """
        Get query execution plan using EXPLAIN (no execution).
        Args:
            query: Cypher query string
        Returns:
            Query plan as dictionary
        """
        with self.driver.session() as session:
            result = session.run(cypher(f'EXPLAIN {query}'))
            plan = result.consume().plan
            assert plan is not None, 'Failed to retrieve query plan.'

            return plan

    def measure_query(self, query: str, num_runs: int) -> tuple[float, list[float], int]:
        """
        Execute a query multiple times and return average execution time.
        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging
        Returns:
            Tuple of (mean_time_ms, times_ms, num_results)
        """
        times_ms = []
        num_results = -1

        with self.driver.session() as session:
            for _ in range(num_runs):
                start = time.perf_counter()
                result = session.run(cypher(query))
                num_results = len(result.data())
                result.consume()  # Ensure full execution
                elapsed_ms = time_quantity.to_base(time.perf_counter() - start, 's')
                times_ms.append(elapsed_ms)

        return np.mean(times_ms).item(), times_ms, num_results

    def get_plan_statistics(self, plans: list[dict]) -> dict[str, Any]:
        """
        Compute statistics about the collected plans.
        Args:
            plans: List of query plans
        Returns:
            Dictionary with statistics
        """
        operator_counts = {}
        total_operators = 0
        max_depth = 0

        def analyze_node(node, depth=0):
            nonlocal total_operators, max_depth

            total_operators += 1
            max_depth = max(max_depth, depth)

            op_type = FeatureExtractor.get_node_type(node)
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1

            for child in FeatureExtractor.get_node_children(node):
                analyze_node(child, depth + 1)

        for plan in plans:
            max_depth = 0  # Reset for each plan
            analyze_node(plan)

        return {
            'total_operators': total_operators,
            'unique_operators': len(operator_counts),
            'operator_counts': operator_counts,
            'max_plan_depth': max_depth,
            'avg_operators_per_plan': total_operators / len(plans) if plans else 0
        }
