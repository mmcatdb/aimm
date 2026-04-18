import time
from typing_extensions import override
from core.drivers import Neo4jDriver, cypher
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

class PlanExtractor(BasePlanExtractor[str]):
    """
    Extract query execution plans and execution times from Neo4j.
    This module handles:
    - Executing queries with EXPLAIN (for plan) and timing actual execution
    - Parsing the plan structure
    - Recording ground truth latencies
    """
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    @override
    def explain_query(self, query: str, do_profile: bool) -> dict:
        # FIXME do_profile

        with self.driver.session() as session:
            result = session.run(cypher(f'EXPLAIN {query}'))
            plan = result.consume().plan
            assert plan is not None, 'Failed to retrieve query plan.'

            return plan

    @override
    def measure_query(self, query: str) -> tuple[float, int]:
        with self.driver.session() as session:
            start = time.perf_counter()
            result = session.run(cypher(query))
            num_results = len(result.data())
            result.consume() # Ensure full execution
            elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

            return elapsed, num_results

    @override
    def collect_global_stats(self) -> dict:
        return {}
