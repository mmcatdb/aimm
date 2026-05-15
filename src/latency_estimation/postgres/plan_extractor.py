import time
from typing_extensions import override
from core.drivers import PostgresDriver
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

class PlanExtractor(BasePlanExtractor[str]):
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, driver: PostgresDriver):
        self.driver = driver

    EXECUTION_TIME_KEY = '$executionTime'

    @override
    def explain_query(self, query: str, is_write: bool, do_profile: bool) -> dict:
        do_rollback = is_write and do_profile
        connection = self.driver.get_connection()
        # Set autocommit to avoid transaction block issues
        connection.autocommit = not do_rollback

        try:
            with connection.cursor() as cursor:
                analyze_option = 'ANALYZE, ' if do_profile else ''
                cursor.execute(f'EXPLAIN ({analyze_option}FORMAT JSON, BUFFERS, VERBOSE) {query}')
                result = cursor.fetchone()

            assert result is not None, 'No plan returned from EXPLAIN.'

            # EXPLAIN returns list of plans
            explain = result[0][0]
            plan: dict = explain['Plan']
            # FIXME This is not used anywhere right now.
            plan[self.EXECUTION_TIME_KEY] = explain['Execution Time']

            return plan
        finally:
            if do_rollback:
                connection.rollback()

            self.driver.put_connection(connection)

    @override
    def measure_query(self, query: str, is_write: bool) -> tuple[float, int]:
        connection = self.driver.get_connection()
        connection.autocommit = not is_write

        try:
            with connection.cursor() as cursor:
                start = time.perf_counter()
                cursor.execute(query)
                if is_write:
                    num_results = cursor.rowcount
                else:
                    results = cursor.fetchall()
                    num_results = len(results)
                elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

            return elapsed, num_results
        finally:
            if is_write:
                connection.rollback()

            self.driver.put_connection(connection)

    @override
    def collect_global_stats(self) -> dict:
        return {}
