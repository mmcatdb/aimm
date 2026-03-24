import time
from typing_extensions import override
import numpy as np
from common.drivers import PostgresDriver
from common.utils import ProgressTracker, print_warning, time_quantity
from common.query_registry import QueryDefMap
from latency_estimation.common import ArrayDataset
from latency_estimation.feature_extractor import BaseDatasetItem

class PostgresItem(BaseDatasetItem):
    def __init__(self, query: str, plan: dict, execution_time: float):
        super().__init__(plan, execution_time)
        self.query = query

    @override
    def query_string(self) -> str:
        return self.query

class PlanExtractor:
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, driver: PostgresDriver):
        self.driver = driver

    def create_dataset(self, queries: list[str], num_runs: int, clear_cache: bool = True, def_map: QueryDefMap[str] | None = None) -> ArrayDataset[PostgresItem]:
        """
        Collect a dataset of query plans and execution times.
        Args:
            queries: List of SQL queries to collect plans for
            num_runs: Number of executions per query for averaging
            clear_cache: Whether to clear cache before each query (slower but more realistic)
        """
        if not clear_cache:
            print('Note: Cache clearing is disabled for faster collection.')
            print('      Set clear_cache=True for cold-cache measurements.\n')

        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        items: list[PostgresItem] = []

        for i, query in enumerate(queries):
            try:
                plan, _ = self.explain_plan(query, clear_cache=clear_cache)
                execution_time, _, _ = self.measure_query(query, num_runs)
                items.append(PostgresItem(query, plan, execution_time))
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

    def explain_plan(self, query: str, clear_cache: bool = True) -> tuple[dict, float]:
        """
        Get the query plan using EXPLAIN without executing the query.
        Args:
            query: SQL query string
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

                # Get the plan without execution (no ANALYZE)
                cursor.execute(f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}')
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                # EXPLAIN returns list of plans
                plan_json = result[0][0]
                return plan_json['Plan'], plan_json['Execution Time']
        finally:
            self.driver.put_connection(connection)

    def measure_query(self, query: str, num_runs: int) -> tuple[float, list[float], int]:
        """
        Execute query and measure actual wall-clock time. Runs multiple times and returns statistics.
        Args:
            query: SQL query string
            num_runs: Number of times to execute the query
        Returns:
            Tuple of (mean_time_ms, times_ms, num_results)
        """
        times_ms = []
        num_results = -1

        for _ in range(num_runs):
            connection = self.driver.get_connection()
            connection.autocommit = True

            try:
                with connection.cursor() as cursor:
                    start = time.perf_counter()
                    cursor.execute(query)
                    # Fetch all results to ensure query completes
                    results = cursor.fetchall()
                    elapsed_ms = time_quantity.to_base(time.perf_counter() - start, 's')
                    times_ms.append(elapsed_ms)
                    num_results = len(results)
            finally:
                self.driver.put_connection(connection)

        return np.mean(times_ms).item(), times_ms, num_results
