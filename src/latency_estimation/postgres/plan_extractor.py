import numpy as np
import time
from common.database import Database
from common.drivers import PostgresDriver
from latency_estimation.postgres.trainer import PostgresDataset
from common.utils import ProgressTracker, print_warning

class PlanExtractor:
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, driver: PostgresDriver, database: Database[str]):
        self.driver = driver
        self.database = database

    def collect_training_dataset(self, num_queries: int, num_runs: int, clear_cache: bool = True) -> PostgresDataset:
        """
        Collect a dataset of query plans and execution times.
        Args:
            num_queries: Number of queries to collect
            num_runs: Number of executions per query for averaging
            clear_cache: Whether to clear cache before each query (slower but more realistic)
        """
        queries = self.database.get_train_queries(num_queries)

        if not clear_cache:
            print('Note: Cache clearing is disabled for faster collection.')
            print('      Set clear_cache=True for cold-cache measurements.\n')

        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        final_queries = []
        plans = []
        execution_times = []

        for i, query in enumerate(queries):
            try:
                plan, execution_time = self.explain_plan_and_measure(query, clear_cache=clear_cache)
                final_queries.append(query)
                plans.append(plan)
                execution_times.append(execution_time)
                progress.track()

            except Exception as e:
                print_warning(f'Could not execute query on index {i}.', e)

        progress.finish()

        dataset = PostgresDataset(final_queries, plans, execution_times)

        print(f'\nCollected {len(dataset)} query plans.')
        return dataset

    def explain_plan_and_measure(self, query: str, clear_cache: bool = True) -> tuple[dict, float]:
        """
        Execute a query and return its plan and actual execution time in ms.
        Args:
            query: SQL query to execute
            num_runs: Number of times to execute for averaging
            clear_cache: Whether to clear PostgreSQL cache before execution
        Returns:
            Tuple of (plan_dict, execution_time_ms)
        """
        connection = self.driver.get_connection()
        # Set autocommit to avoid transaction block issues
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                # Clear cache if requested (simulates cold cache)
                if clear_cache:
                    try:
                        cursor.execute('DISCARD ALL')
                    except Exception as e:
                        # NICE_TO_HAVE If DISCARD ALL fails, try alternative cache clearing
                        print_warning(f'Could not clear cache.', e)
                        pass

                # Get the plan with execution statistics
                cursor.execute(f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}')
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                # EXPLAIN returns list of plans
                plan_json = result[0][0]
                return plan_json['Plan'], plan_json['Execution Time']
        finally:
            self.driver.put_connection(connection)

    def explain_plan(self, query: str) -> dict:
        """
        Get the query plan using EXPLAIN without executing the query.
        Args:
            query: SQL query string
        Returns:
            Query plan dictionary
        """
        connection = self.driver.get_connection()
        # Set autocommit to avoid transaction block issues
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                # Get the plan without execution (no ANALYZE)
                cursor.execute(f'EXPLAIN (FORMAT JSON, VERBOSE) {query}')
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                # EXPLAIN returns list of plans
                return result[0][0]['Plan']
        finally:
            self.driver.put_connection(connection)

    def measure_query(self, query: str, num_runs: int) -> tuple[float, float, float, int]:
        """
        Execute query and measure actual wall-clock time. Runs multiple times and returns statistics.
        Args:
            query: SQL query string
            num_runs: Number of times to execute the query
        Returns:
            Tuple of (mean_time_ms, min_time_ms, max_time_ms, num_results)
        """
        times_ms = []
        num_results = -1

        for _ in range(num_runs):
            connection = self.driver.get_connection()
            connection.autocommit = True

            try:
                with connection.cursor() as cursor:
                    start_s = time.time()
                    cursor.execute(query)
                    # Fetch all results to ensure query completes
                    results = cursor.fetchall()
                    end_s = time.time()

                    times_ms.append((end_s - start_s) * 1000)
                    num_results = len(results)
            finally:
                self.driver.put_connection(connection)

        return np.mean(times_ms).item(), np.min(times_ms), np.max(times_ms), num_results
