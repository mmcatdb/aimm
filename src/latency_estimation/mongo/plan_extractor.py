import time
from typing_extensions import override
from core.query import MongoQuery, MongoFindQuery, MongoAggregateQuery
from core.drivers import MongoDriver
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

class PlanExtractor(BasePlanExtractor[MongoQuery]):
    """Extracts query plans and execution statistics from MongoDB."""

    def __init__(self, driver: MongoDriver):
        self.config = driver
        self.db = driver.database()
        self._collection_stats_cache = dict[str, dict]()

    @override
    def explain_query(self, query: MongoQuery, do_profile: bool) -> dict:
        if isinstance(query, MongoFindQuery):
            return self.__explain_find(query, do_profile)
        elif isinstance(query, MongoAggregateQuery):
            return self.__explain_aggregate(query, do_profile)
        else:
            raise ValueError(f'Unsupported query type for explain: {type(query)}')

    def __explain_find(self, query: MongoFindQuery, do_profile: bool) -> dict:
        cmd = query.to_dict()
        return self.__explain_common(cmd, do_profile, query.collection)

    def __explain_aggregate(self, query: MongoAggregateQuery, do_profile: bool) -> dict:
        cmd = query.to_dict()
        cmd['cursor'] = {} # Yes, this is required.
        return self.__explain_common(cmd, do_profile, query.collection)

    GLOBAL_STATS_COLLECTION_KEY = '$collection'

    def __explain_common(self, cmd: dict, do_profile: bool, collection: str) -> dict:
        verbosity = 'executionStats' if do_profile else 'queryPlanner'
        explain = self.db.command('explain', cmd, verbosity=verbosity)

        # FIXME This is probably not correct. The winning plan is just a small part of the explain output. Moreover, the explain output for aggregation pipelines contains multiple stages from which only the first one is currently extracted. Explain doesn't even provide plans for the other stages ...
        # We whould probably create our own plan structure. Some of the stages can be further explored by running explain on them (resp. their corresponding queries) separately.

        plan = self.__extract_plan(explain)
        # Add collection name to the root node.
        plan[self.GLOBAL_STATS_COLLECTION_KEY] = collection

        return plan

    def __extract_plan(self, explain_output: dict) -> dict:
        """Extracts the plan tree from an explain output.

        Handles both explainVersion 1 (classic) and 2 (SBE / aggregation).
        """
        # The output we are interested in might not depend on the version?
        # version = explain_output.get('explainVersion', '1')

        stage = self.__extract_plan_stage(explain_output)
        # Slot-based Execution Engine will have `queryPlan`. Otherwise, it should be the winning plan directly.
        wp = stage['queryPlanner']['winningPlan']
        return wp.get('queryPlan', wp)

    def __extract_plan_stage(self, explain_output: dict) -> dict:
        if 'stages' in explain_output:
            # Aggregate: may have stages[] with $cursor
            return explain_output['stages'][0]['$cursor']

        return explain_output

    @override
    def measure_query(self, query: MongoQuery) -> tuple[float, int]:
        # FIXME Rollback if it's a write query.

        if isinstance(query, MongoFindQuery):
            return self.__measure_find(query)
        elif isinstance(query, MongoAggregateQuery):
            return self.__measure_aggregate(query)
        else:
            # FIXME Implement other types.
            raise ValueError(f'Unsupported query type for measurement: {type(query)}')

    def __measure_find(self, query: MongoFindQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        if query.projection:
            cursor = collection.find(query.filter, query.projection)
        else:
            cursor = collection.find(query.filter)

        if query.sort:
            cursor = cursor.sort(list(query.sort.items()))
        if query.skip:
            cursor = cursor.skip(query.skip)
        if query.limit:
            cursor = cursor.limit(query.limit)

        start = time.perf_counter()
        results = list(cursor)  # force materialization
        elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed, len(results)

    def __measure_aggregate(self, query: MongoAggregateQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        start = time.perf_counter()
        results = list(collection.aggregate(query.pipeline))
        elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed, len(results)

    @override
    def collect_global_stats(self) -> dict:
        stats = dict[str, dict]()
        for collection in self.db.list_collection_names():
            if not collection.startswith("system."):
                stats[collection] = self.__fetch_global_stats(collection)

        return stats

    def __fetch_global_stats(self, collection: str) -> dict:
        stats = self.db.command("collStats", collection)
        return {
            'count': stats.get('count', 0),
            'size': stats.get('size', 0),
            'avgObjSize': stats.get('avgObjSize', 0),
            'storageSize': stats.get('storageSize', 0),
            'nindexes': stats.get('nindexes', 0),
            'totalIndexSize': stats.get('totalIndexSize', 0),
        }
