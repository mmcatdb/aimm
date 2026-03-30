import time
from typing_extensions import override
from common.drivers import PostgresDriver
from common.utils import ProgressTracker, print_warning, time_quantity
from common.query_registry import QueryDefMap
from latency_estimation.common import ArrayDataset
from latency_estimation.feature_extractor import BaseDatasetItem
from latency_estimation.plan_extractor import BasePlanExtractor

class PostgresItem(BaseDatasetItem):
    def __init__(self, id: str, query: str, plan: dict, times: list[float]):
        super().__init__(id, plan, times)
        self.query = query

    @override
    def query_string(self) -> str:
        return self.query

class PlanExtractor(BasePlanExtractor[str]):
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, driver: PostgresDriver):
        self.driver = driver

    def create_dataset(self, queries: list[str], num_runs: int, def_map: QueryDefMap[str], clear_cache: bool = True) -> ArrayDataset[PostgresItem]:
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

        items = list[PostgresItem]()

        for i, query in enumerate(queries):
            try:
                plan, _ = self.explain_query(query, clear_cache=clear_cache)
                times = self.measure_query_multiple(query, num_runs)
                items.append(PostgresItem(def_map[id(query)].id, query, plan, times))
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

    def explain_query(self, query: str, clear_cache: bool = True) -> tuple[dict, float]:
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

                # Get the plan without execution (no ANALYZE)
                cursor.execute(f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}')
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                # EXPLAIN returns list of plans
                plan_json = result[0][0]
                return plan_json['Plan'], plan_json['Execution Time']
        finally:
            self.driver.put_connection(connection)

    @override
    def measure_query(self, query: str) -> tuple[float, int]:
        connection = self.driver.get_connection()
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                start = time.perf_counter()
                cursor.execute(query)
                # Fetch all results to ensure query completes
                results = cursor.fetchall()
                elapsed_ms = time_quantity.to_base(time.perf_counter() - start, 's')
                num_results = len(results)

                return elapsed_ms, num_results
        finally:
            self.driver.put_connection(connection)
