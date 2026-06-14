import time
import re
from typing import Any
from neo4j import Session, Transaction
from typing_extensions import override
from core.drivers import Neo4jDriver, cypher
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

DML_RE = re.compile(r'^\s*(CREATE|DELETE|DETACH\s+DELETE|SET|REMOVE|MERGE)\b', re.IGNORECASE | re.MULTILINE)

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
    def explain_query(self, query: str, is_write: bool, do_profile: bool) -> dict:
        with self.driver.session() as session:
            if do_profile:
                if is_write:
                    tx = session.begin_transaction()
                    try:
                        result = tx.run(cypher(f'PROFILE {query}'))
                        for _ in result:
                            pass
                        summary = result.consume()
                        profile = summary.profile
                        assert profile is not None, 'Failed to retrieve query profile.'
                        return _normalize_plan(profile, include_profile=True)
                    finally:
                        tx.rollback()

                result = session.run(cypher(f'PROFILE {query}'))
                for _ in result:
                    pass
                summary = result.consume()
                profile = summary.profile
                assert profile is not None, 'Failed to retrieve query profile.'
                return _normalize_plan(profile, include_profile=True)

            result = session.run(cypher(f'EXPLAIN {query}'))
            plan = result.consume().plan
            assert plan is not None, 'Failed to retrieve query plan.'

            return _normalize_plan(plan, include_profile=False)

    @override
    def measure_query(self, query: str, is_write: bool) -> tuple[float, int]:
        with self.driver.session() as session:
            if is_write:
                tx = session.begin_transaction()
                try:
                    return self.__measure_inner(tx, query)
                finally:
                    tx.rollback()
            else:
                return self.__measure_inner(session, query)

    def __measure_inner(self, session_or_tx: Session | Transaction, query: str) -> tuple[float, int]:
        start = time.perf_counter()
        result = session_or_tx.run(cypher(query))
        num_results = len(result.data())
        result.consume()  # Ensure full execution
        elapsed = time_quantity.to_base(time.perf_counter() - start, 's')
        return elapsed, num_results

    @override
    def collect_global_stats(self) -> dict:
        return {}


def _normalize_plan(plan: dict, include_profile: bool) -> dict[str, Any]:
    if not plan:
        return {}

    args = dict(plan.get('args') or plan.get('arguments') or {})
    output: dict[str, Any] = {
        'operatorType': plan.get('operatorType', 'Unknown'),
        'args': args,
        'identifiers': list(plan.get('identifiers') or []),
        'children': [_normalize_plan(child, include_profile) for child in plan.get('children', [])],
    }

    if include_profile:
        output['profile'] = {
            'dbHits': plan.get('dbHits', args.get('DbHits')),
            'rows': plan.get('rows', args.get('Rows')),
            'pageCacheHits': plan.get('pageCacheHits', args.get('PageCacheHits')),
            'pageCacheMisses': plan.get('pageCacheMisses', args.get('PageCacheMisses')),
            'time': plan.get('time', args.get('Time')),
        }

    return output
