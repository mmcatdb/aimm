"""
Extract query execution plans and execution times from Neo4j.
This module handles:
- Generating TPC-H query variants with random parameters
- Executing queries with EXPLAIN (for plan) and timing actual execution
- Parsing the plan structure
- Recording ground truth latencies
"""
import time
from typing import Any
import numpy as np

from datasets.database import Database
from common.drivers import Neo4jDriver, cypher

class PlanExtractor:
    """
    Extracts query plans and execution times from Neo4j.
    Generates multiple query variants by substituting parameters.
    """

    NUM_QUERY_TYPES = 32

    def __init__(self, neo4j: Neo4jDriver, database: Database):
        self.neo4j = neo4j
        self.database = database

    def close(self):
        """Close database connection."""
        self.neo4j.close()

    def get_plan(self, query: str) -> dict:
        """
        Get query execution plan using EXPLAIN (no execution).

        Args:
            query: Cypher query string

        Returns:
            Query plan as dictionary
        """
        with self.neo4j.session() as session:
            result = session.run(cypher(f'EXPLAIN {query}'))
            plan = result.consume().plan
            assert plan is not None, 'Failed to retrieve query plan.'

            return plan

    def execute_query(self, query: str, num_runs: int = 1, show_details: bool = False) -> float:
        """
        Execute a query multiple times and return average execution time.

        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging

        Returns:
            Average execution time in seconds
        """
        execution_times = []

        with self.neo4j.session() as session:
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(cypher(query))

                if show_details:
                    print(query)
                    print()
                    # print('Result sample:')
                    print(result.data())
                    print('-'*40)

                result.consume()  # Ensure full execution
                end_time = time.time()
                execution_times.append(end_time - start_time)

        return np.mean(execution_times).item()

    def get_plan_and_execute(self, query: str, num_runs: int = 1, show_details: bool = False) -> tuple[dict, float]:
        """
        Get query plan with EXPLAIN and measure actual execution time.

        Args:
            query: Cypher query string
            num_runs: Number of times to execute for averaging

        Returns:
            Tuple of (plan, average_execution_time_seconds)
        """
        plan = self.get_plan(query)

        # Measure actual execution time
        execution_time = self.execute_query(query, num_runs, show_details=show_details)

        return plan, execution_time

    def collect_workload(self, num_queries: int = 350, num_runs_per_query: int = 1, show_details: bool = False) -> tuple[list[str], list[dict], list[float]]:
        """
        Collect a workload of query plans and execution times.

        Args:
            num_queries: Total number of queries to generate
            num_runs_per_query: Number of executions per query for averaging

        Returns:
            Tuple of (queries, plans, execution_times)
        """
        # Generate queries
        queries = self.database.get_train_queries(num_queries)

        print(f'\nExecuting {len(queries)} queries...')
        print(f'Each query will be executed {num_runs_per_query} times for averaging.\n')

        all_plans = []
        all_times = []

        for i, query in enumerate(queries):
            try:
                # Get plan and execution time
                plan, exec_time = self.get_plan_and_execute(query, num_runs_per_query, show_details=show_details)

                all_plans.append(plan)
                all_times.append(exec_time)

                if i % 100 == 0 and i > 0:
                    print(f'Extracted {i} / {len(queries)} plans...')

            except Exception as e:
                print(f' ERROR: {str(e)}')
                continue

        print(f'\n{"=" * 60}')
        print(f'Workload collection complete!')
        print(f'  Total queries: {len(queries)}')
        print(f'  Average execution time: {np.mean(all_times):.4f}s')
        print(f'  Min/Max execution time: {np.min(all_times):.4f}s / {np.max(all_times):.4f}s')
        print(f'{"=" * 60}\n')

        return queries, all_plans, all_times

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

    def collect_training_data(self, queries: list[str], num_runs: int = 3) -> tuple[list[str], list[dict], list[float]]:
        """
        Collect training data by executing queries and measuring their performance.

        Args:
            queries: List of Cypher query strings
            num_runs: Number of executions per query for averaging

        Returns:
            Tuple of (queries, plans, execution_times)
        """
        print(f'Collecting training data from {len(queries)} queries...')
        print(f'Each query will be executed {num_runs} times for averaging.')

        collected_queries = []
        plans = []
        execution_times = []

        for i, query in enumerate(queries):
            try:
                print(f'\nQuery {i+1}/{len(queries)}:')
                print(f'  {query[:100]}...' if len(query) > 100 else f'  {query}')

                # Get plan and execution time
                plan, exec_time = self.get_plan_and_execute(query, num_runs)

                collected_queries.append(query)
                plans.append(plan)
                execution_times.append(exec_time)

                print(f'  Execution time: {exec_time:.4f}s')
                print(f'  Root operator: {plan.get("operatorType", "Unknown")}')

            except Exception as e:
                print(f'  ERROR: Failed to process query: {e}')
                continue

        print(f'\nSuccessfully collected {len(plans)} query plans.')
        return collected_queries, plans, execution_times
