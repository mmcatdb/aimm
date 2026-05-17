import time
from pymongo.collection import Collection
import re
from typing_extensions import override
from core.query import MongoQuery, MongoFindQuery, MongoAggregateQuery, MongoUpdateQuery, MongoDeleteQuery, MongoInsertQuery
from core.drivers import MongoDriver
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

class PlanExtractor(BasePlanExtractor[MongoQuery]):
    """Extracts query plans and execution statistics from MongoDB."""

    def __init__(self, driver: MongoDriver):
        self.driver = driver
        self.db = driver.database()
        self._collection_stats_cache = dict[str, dict]()

    @override
    def explain_query(self, query: MongoQuery, is_write: bool, do_profile: bool) -> dict:
        if isinstance(query, (
            MongoFindQuery,
            MongoUpdateQuery,
            MongoDeleteQuery,
        )):
            cmd = query.to_dict()
            return self.__explain_common(cmd, do_profile, query.collection)
        elif isinstance(query, MongoAggregateQuery):
            return self.__explain_aggregate(query, do_profile)
        elif isinstance(query, MongoInsertQuery):
            # MongoDB does not support explain for insert commands.
            return {'stage': 'INSERT', self.GLOBAL_STATS_COLLECTION_KEY: query.collection}
        else:
            raise ValueError(f'Unsupported query type for explain: {type(query)}')

    def __explain_aggregate(self, query: MongoAggregateQuery, do_profile: bool) -> dict:
        cmd = query.to_dict()
        cmd['cursor'] = {} # Yes, this is required.
        cmd['allowDiskUse'] = True
        return self.__explain_common(cmd, do_profile, query.collection)

    GLOBAL_STATS_COLLECTION_KEY = '$collection'

    def __explain_common(self, cmd: dict, do_profile: bool, collection: str) -> dict:
        verbosity = 'executionStats' if do_profile else 'queryPlanner'
        explain = self.db.command('explain', cmd, verbosity=verbosity)

        plan = self.__extract_plan(explain)
        # Add collection name to the root node.
        plan[self.GLOBAL_STATS_COLLECTION_KEY] = collection

        return plan

    def __extract_plan(self, explain_output: dict) -> dict:
        """Extracts the plan tree from an explain output.

        Handles both explainVersion 1 (classic) and 2 (SBE / aggregation).
        """
        if 'stages' in explain_output:
            return self.__extract_aggregation_plan(explain_output)

        stage = self.__extract_plan_stage(explain_output)
        if stage is None:
            return self.__fallback_plan(explain_output)

        query_planner = stage.get('queryPlanner')
        if not isinstance(query_planner, dict):
            return self.__fallback_plan(explain_output)

        winning_plan = query_planner.get('winningPlan')
        if not isinstance(winning_plan, dict):
            return self.__fallback_plan(explain_output)

        # Slot-based Execution Engine will have `queryPlan`. Otherwise, it should be the winning plan directly.
        return winning_plan.get('queryPlan', winning_plan)

    def __extract_plan_stage(self, explain_output: dict) -> dict | None:
        if 'stages' in explain_output:
            # Aggregations usually include a $cursor stage, but it is not
            # guaranteed to be the first stage for every server/version/pipeline.
            for stage in explain_output['stages']:
                if '$cursor' in stage:
                    return stage['$cursor']

            return None

        return explain_output

    def __extract_aggregation_plan(self, explain_output: dict) -> dict:
        cursor_stage_index = None
        cursor_stage = None
        for index, stage in enumerate(explain_output.get('stages', [])):
            if '$cursor' in stage:
                cursor_stage_index = index
                cursor_stage = stage['$cursor']
                break

        if cursor_stage is None or cursor_stage_index is None:
            return self.__fallback_plan(explain_output)

        query_planner = cursor_stage.get('queryPlanner')
        if not isinstance(query_planner, dict):
            return self.__fallback_plan(explain_output)

        winning_plan = query_planner.get('winningPlan')
        if not isinstance(winning_plan, dict):
            return self.__fallback_plan(explain_output)

        current = winning_plan.get('queryPlan', winning_plan)
        for stage in explain_output.get('stages', [])[cursor_stage_index + 1:]:
            current = self.__aggregation_stage_to_plan_node(stage, current)

        return current

    def __aggregation_stage_to_plan_node(self, stage: dict, input_stage: dict) -> dict:
        if not isinstance(stage, dict) or len(stage) != 1:
            return {
                'stage': 'AGG_UNKNOWN',
                'rawStageKeys': list(stage.keys()) if isinstance(stage, dict) else [],
                'inputStage': input_stage,
            }

        operator, spec = next(iter(stage.items()))
        node = {
            'stage': self.__aggregation_stage_name(operator),
            'inputStage': input_stage,
        }

        if operator == '$match':
            node['filter'] = spec
        elif operator in ('$project', '$addFields', '$set', '$unset'):
            node['transformBy'] = spec
        elif operator == '$sort':
            node['sortPattern'] = spec.get('sortKey', spec) if isinstance(spec, dict) else {}
        elif operator == '$limit':
            node['limitAmount'] = spec
        elif operator == '$skip':
            node['skipAmount'] = spec
        elif operator == '$group':
            if isinstance(spec, dict):
                node['groupBy'] = spec.get('_id')
                node['accumulators'] = {key: value for key, value in spec.items() if key != '_id'}
        elif operator == '$lookup' and isinstance(spec, dict):
            node['foreignCollection'] = spec.get('from', '')
            if 'pipeline' in spec:
                node['pipelineStageCount'] = len(spec.get('pipeline') or [])
        elif operator == '$unwind':
            node['path'] = spec.get('path') if isinstance(spec, dict) else spec
        elif operator == '$count':
            node['countField'] = spec

        return node

    def __aggregation_stage_name(self, operator: str) -> str:
        raw = operator.strip('$').replace('-', '_')
        normalized = re.sub(r'(?<!^)(?=[A-Z])', '_', raw).upper()
        return f'AGG_{normalized or "UNKNOWN"}'

    def __fallback_plan(self, explain_output: dict) -> dict:
        """Return a minimal plan instead of crashing on valid but unfamiliar explain shapes."""
        return {
            'stage': 'UNKNOWN',
            'explainVersion': explain_output.get('explainVersion'),
        }

    @override
    def measure_query(self, query: MongoQuery, is_write: bool) -> tuple[float, int]:
        # Event though we don't need the `is_write` argument, we use it to force consistent usage of this flag for mongo.
        if is_write:
            if isinstance(query, MongoUpdateQuery):
                return self.__measure_update(query)
            elif isinstance(query, MongoDeleteQuery):
                return self.__measure_delete(query)
            elif isinstance(query, MongoInsertQuery):
                return self.__measure_insert(query)
        else:
            if isinstance(query, MongoFindQuery):
                return self.__measure_find(query)
            elif isinstance(query, MongoAggregateQuery):
                return self.__measure_aggregate(query)

        raise ValueError(f'Unsupported query type for measurement: {type(query)}')

    def __measure_find(self, query: MongoFindQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        filter_doc = query.filter or {}
        if query.projection is not None:
            cursor = collection.find(filter_doc, query.projection)
        else:
            cursor = collection.find(filter_doc)

        if query.sort is not None:
            cursor = cursor.sort(list(query.sort.items()))
        if query.skip is not None:
            cursor = cursor.skip(query.skip)
        if query.limit is not None:
            cursor = cursor.limit(query.limit)

        start = time.perf_counter()
        num_results = self.__consume_cursor(cursor)
        elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed, num_results

    def __measure_aggregate(self, query: MongoAggregateQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        start = time.perf_counter()
        num_results = self.__consume_cursor(collection.aggregate(query.pipeline, allowDiskUse=True))
        elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed, num_results

    @staticmethod
    def __consume_cursor(cursor) -> int:
        num_results = 0
        for _ in cursor:
            num_results += 1
        return num_results

    def __measure_update(self, query: MongoUpdateQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        original_docs = self.__get_original_docs(collection, query.filter, query.multi)
        try:
            start = time.perf_counter()
            result = collection.update_many(query.filter, query.update) if query.multi else collection.update_one(query.filter, query.update)
            elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

            return elapsed, result.modified_count
        finally:
            for doc in original_docs:
                collection.replace_one({'_id': doc['_id']}, doc)

    def __measure_delete(self, query: MongoDeleteQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        original_docs = self.__get_original_docs(collection, query.filter, query.multi)
        try:
            start = time.perf_counter()
            result = collection.delete_many(query.filter) if query.multi else collection.delete_one(query.filter)
            elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

            return elapsed, result.deleted_count
        finally:
            if original_docs:
                collection.insert_many(original_docs)

    def __get_original_docs(self, collection: Collection, filter: dict, is_multi: bool) -> list[dict]:
        if is_multi:
            return list(collection.find(filter))

        doc = collection.find_one(filter)
        return [doc] if doc is not None else []

    def __measure_insert(self, query: MongoInsertQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        inserted_ids = []
        try:
            start = time.perf_counter()
            result = collection.insert_many(query.documents)
            inserted_ids = result.inserted_ids
            elapsed = time_quantity.to_base(time.perf_counter() - start, 's')

            return elapsed, len(inserted_ids)
        finally:
            if inserted_ids:
                collection.delete_many({'_id': {'$in': inserted_ids}})

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
