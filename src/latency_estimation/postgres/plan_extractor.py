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
    def explain_query(self, query: str, do_profile: bool) -> dict:
        # FIXME do_profile

        connection = self.driver.get_connection()
        # Set autocommit to avoid transaction block issues
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                # Get the plan without execution (no ANALYZE)
                cursor.execute(f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}')
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                # EXPLAIN returns list of plans
                explain = result[0][0]
                plan: dict = explain['Plan']
                # FIXME This is not used anywhere right now.
                plan[self.EXECUTION_TIME_KEY] = explain['Execution Time']

                return plan
        finally:
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

            if is_write:
                connection.rollback()

            return elapsed, num_results
        except:
            if is_write:
                connection.rollback()
            raise
        finally:
            self.driver.put_connection(connection)

    @override
    def collect_global_stats(self) -> dict:
        return {}
