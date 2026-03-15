import time
from typing import Any
import numpy as np
from common.database import Database
from common.drivers import Neo4jDriver, cypher
from latency_estimation.abstract import BaseDataset
from common.utils import ProgressTracker

class PlanExtractor:
    """
    Extract query execution plans and execution times from Neo4j.
    This module handles:
    - Executing queries with EXPLAIN (for plan) and timing actual execution
    - Parsing the plan structure
    - Recording ground truth latencies
    """
    def __init__(self, driver: Neo4jDriver, database: Database[str]):
        self.driver = driver
        self.database = database

    def collect_training_dataset(self, num_queries: int, num_runs: int, show_details: bool = False) -> BaseDataset[str]:
        """
        Collect a workload of query plans and execution times.
        Args:
            num_queries: Total number of queries to generate
            num_runs: Number of executions per query for averaging
        """
        queries = self.database.get_train_queries(num_queries)

        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        plans = []
        execution_times = []

        for i, query in enumerate(queries):
            try:
                # Get plan and execution time
                plan, execution_time = self.__explain_plan_and_measure(query, num_runs, show_details=show_details)

                plans.append(plan)
                execution_times.append(execution_time)
                progress.track()

            except Exception as e:
                print(f'\nError executing query {i}: {e}')
                print(f'Query preview: {query[:100]}...\n')
                continue

        progress.finish()

        dataset = BaseDataset(queries, plans, execution_times)

        print(f'\nCollected {len(dataset)} query plans.')
        return dataset

    def __explain_plan_and_measure(self, query: str, num_runs: int, show_details: bool = False) -> tuple[dict, float]:
        """
        Get query plan with EXPLAIN and measure actual execution time.
        Args:
            query: Cypher query string
            num_runs: Number of times to execute for averaging
        Returns:
            Tuple of (plan, average_execution_time_seconds)
        """
        plan = self.explain_plan(query)
        execution_time = self.measure_query(query, num_runs, show_details=show_details)[0]

        return plan, execution_time

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

    def measure_query(self, query: str, num_runs: int, show_details: bool = False) -> tuple[float, float, int]:
        """
        Execute a query multiple times and return average execution time.
        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging
        Returns:
            Tuple of (mean_time_ms, std_time_ms, num_results)
        """
        times_ms = []
        num_results = -1

        with self.driver.session() as session:
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(cypher(query))
                num_results = len(result.data())

                if show_details:
                    print(query)
                    print()
                    # print('Result sample:')
                    print(result.data())
                    print('-' * 40)

                result.consume()  # Ensure full execution
                end_time = time.time()
                times_ms.append(end_time - start_time)

        return np.mean(times_ms).item(), np.std(times_ms).item(), num_results

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

            op_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1

            for child in node.get('children', []):
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
