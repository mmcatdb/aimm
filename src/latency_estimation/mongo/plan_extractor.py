from __future__ import annotations
import time
from pymongo.collection import Collection
import re
from datetime import date, datetime
from typing import Any
from typing_extensions import override
from core.query import MongoQuery, MongoFindQuery, MongoAggregateQuery, MongoUpdateQuery, MongoDeleteQuery, MongoInsertQuery
from core.drivers import MongoDriver
from core.utils import time_quantity
from latency_estimation.plan_extractor import BasePlanExtractor

class PlanExtractor(BasePlanExtractor[MongoQuery]):
    """Extracts query plans and execution statistics from MongoDB."""

    FIELD_STATS_VERSION = 1
    GLOBAL_STATS_VERSION_KEY = 'globalStatsVersion'
    FIELD_STATS_KEY = 'fieldStats'
    FIELD_STATS_META_KEY = 'fieldStatsMeta'
    QUERY_FILTER_KEY = '$queryFilter'
    QUERY_FILTERS_KEY = '$queryFilters'

    FIELD_DISCOVERY_SAMPLE_SIZE = 200
    MAX_STATS_FIELDS_PER_COLLECTION = 96
    MAX_ARRAY_VALUES_PER_SAMPLE = 8
    MAX_FIELD_DISCOVERY_DEPTH = 8
    TOP_VALUE_LIMIT = 16
    DISTINCT_COUNT_CAP = 20_000
    HISTOGRAM_BUCKETS = 12
    FIELD_STATS_MAX_TIME_MS = 300_000

    SCALAR_VALUE_TYPES = {'bool', 'date', 'decimal', 'double', 'int', 'long', 'string'}
    NUMERIC_VALUE_TYPES = {'decimal', 'double', 'int', 'long'}
    RANGE_VALUE_TYPES = NUMERIC_VALUE_TYPES | {'date'}

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
        query_filters = self.__extract_query_filters(cmd)
        if query_filters:
            plan[self.QUERY_FILTERS_KEY] = query_filters
            if len(query_filters) == 1:
                plan[self.QUERY_FILTER_KEY] = query_filters[0]

        return plan

    def __extract_query_filters(self, cmd: dict) -> list[dict]:
        """Return the original predicate documents from the command.

        These are queryPlanner-safe because they come from the query text, not
        from executing the query. Keeping them on the plan lets feature
        extraction estimate selectivity from precomputed stats even when MongoDB
        pushes indexed predicates into indexBounds instead of a FETCH filter.
        """
        filters = list[dict]()

        filter_doc = cmd.get('filter')
        if isinstance(filter_doc, dict) and filter_doc:
            filters.append(filter_doc)

        for update in cmd.get('updates') or []:
            query_doc = update.get('q') if isinstance(update, dict) else None
            if isinstance(query_doc, dict) and query_doc:
                filters.append(query_doc)

        for delete in cmd.get('deletes') or []:
            query_doc = delete.get('q') if isinstance(delete, dict) else None
            if isinstance(query_doc, dict) and query_doc:
                filters.append(query_doc)

        pipeline = cmd.get('pipeline')
        if isinstance(pipeline, list):
            for stage in pipeline:
                if not isinstance(stage, dict):
                    continue
                match_doc = stage.get('$match')
                if isinstance(match_doc, dict) and match_doc:
                    filters.append(match_doc)

        return filters

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
        collection_count = stats.get('count', 0)
        index_summary = self.__index_field_summary(collection)
        field_stats = self.__fetch_field_distribution_stats(collection, collection_count, index_summary)
        return {
            'count': stats.get('count', 0),
            'size': stats.get('size', 0),
            'avgObjSize': stats.get('avgObjSize', 0),
            'storageSize': stats.get('storageSize', 0),
            'nindexes': stats.get('nindexes', 0),
            'totalIndexSize': stats.get('totalIndexSize', 0),
            'indexedFields': sorted(index_summary['indexed']),
            'uniqueFields': sorted(index_summary['unique']),
            self.GLOBAL_STATS_VERSION_KEY: self.FIELD_STATS_VERSION,
            self.FIELD_STATS_KEY: field_stats,
            self.FIELD_STATS_META_KEY: {
                'version': self.FIELD_STATS_VERSION,
                'field_count': len(field_stats),
                'sample_size': self.FIELD_DISCOVERY_SAMPLE_SIZE,
                'top_value_limit': self.TOP_VALUE_LIMIT,
                'distinct_count_cap': self.DISTINCT_COUNT_CAP,
                'histogram_buckets': self.HISTOGRAM_BUCKETS,
            },
        }

    def __index_field_summary(self, collection: str) -> dict[str, set[str]]:
        indexed = set[str]()
        unique = set[str]()
        compound = set[str]()

        try:
            index_information = self.db[collection].index_information()
        except Exception:
            index_information = {}

        for info in index_information.values():
            keys = [
                field
                for field, _ in info.get('key', [])
                if isinstance(field, str) and field not in ('_id', '_fts', '_ftsx')
            ]
            if not keys:
                continue

            indexed.update(keys)
            if len(keys) > 1:
                compound.update(keys)
            if info.get('unique') and len(keys) == 1:
                unique.add(keys[0])

        return {
            'indexed': indexed,
            'unique': unique,
            'compound': compound,
        }

    def __fetch_field_distribution_stats(self, collection: str, collection_count: int, index_summary: dict[str, set[str]]) -> dict:
        collection_obj = self.db[collection]
        discovered = self.__discover_sample_field_paths(collection_obj)
        fields = self.__select_distribution_fields(index_summary['indexed'], discovered)

        output = dict[str, dict]()
        for field in fields:
            try:
                output[field] = self.__fetch_field_stats(collection_obj, field, collection_count, index_summary, discovered.get(field, {}))
            except Exception as e:
                output[field] = {
                    'count': collection_count,
                    'error': str(e)[:240],
                    'is_indexed': field in index_summary['indexed'],
                    'is_unique': field in index_summary['unique'],
                }

        return output

    def __discover_sample_field_paths(self, collection: Collection) -> dict[str, dict]:
        metadata = dict[str, dict]()
        try:
            cursor = collection.find({}, limit=self.FIELD_DISCOVERY_SAMPLE_SIZE)
            for document in cursor:
                self.__discover_value_paths(document, '', metadata, 0)
        except Exception:
            return metadata

        return metadata

    def __discover_value_paths(self, value: Any, prefix: str, metadata: dict[str, dict], depth: int) -> None:
        if depth > self.MAX_FIELD_DISCOVERY_DEPTH:
            return

        if isinstance(value, dict):
            for key, child in value.items():
                if key == '_id':
                    continue
                child_path = f'{prefix}.{key}' if prefix else key
                self.__discover_value_paths(child, child_path, metadata, depth + 1)
            return

        if isinstance(value, list):
            if prefix:
                item = metadata.setdefault(prefix, self.__empty_field_discovery())
                item['array_count'] += 1
                item['max_array_length'] = max(item['max_array_length'], len(value))
            for child in value[:self.MAX_ARRAY_VALUES_PER_SAMPLE]:
                self.__discover_value_paths(child, prefix, metadata, depth + 1)
            return

        if not prefix:
            return

        item = metadata.setdefault(prefix, self.__empty_field_discovery())
        item['scalar_count'] += 1
        type_name = self.__sample_type_name(value)
        item['types'].add(type_name)
        if isinstance(value, str):
            item['max_string_length'] = max(item['max_string_length'], len(value))
        if self.__is_hashable_sample_value(value):
            if len(item['distinct_sample']) < 256:
                item['distinct_sample'].add(value)

    def __empty_field_discovery(self) -> dict:
        return {
            'scalar_count': 0,
            'array_count': 0,
            'max_array_length': 0,
            'max_string_length': 0,
            'types': set[str](),
            'distinct_sample': set(),
        }

    def __select_distribution_fields(self, indexed_fields: set[str], discovered: dict[str, dict]) -> list[str]:
        fields = sorted(indexed_fields)
        extra_fields = [
            field
            for field, metadata in discovered.items()
            if field not in indexed_fields and self.__should_collect_sample_field(metadata)
        ]

        remaining = max(0, self.MAX_STATS_FIELDS_PER_COLLECTION - len(fields))
        fields.extend(sorted(extra_fields)[:remaining])
        return fields

    def __should_collect_sample_field(self, metadata: dict) -> bool:
        scalar_count = int(metadata.get('scalar_count', 0))
        array_count = int(metadata.get('array_count', 0))
        if scalar_count <= 0:
            return array_count > 0

        types = metadata.get('types', set())
        if types & {'bool', 'date', 'number'}:
            return True

        if 'string' in types:
            distinct_count = len(metadata.get('distinct_sample', set()))
            max_string_length = int(metadata.get('max_string_length', 0))
            return max_string_length <= 96 and distinct_count <= max(32, scalar_count // 2)

        return False

    def __fetch_field_stats(
        self,
        collection: Collection,
        field: str,
        collection_count: int,
        index_summary: dict[str, set[str]],
        discovery_metadata: dict,
    ) -> dict:
        stats = {
            'count': collection_count,
            'is_indexed': field in index_summary['indexed'],
            'is_unique': field in index_summary['unique'],
            'is_compound_indexed': field in index_summary['compound'],
            'sample_scalar_count': int(discovery_metadata.get('scalar_count', 0) or 0),
            'sample_array_count': int(discovery_metadata.get('array_count', 0) or 0),
        }

        stats.update(self.__fetch_field_presence_stats(collection, field))
        stats.update(self.__fetch_field_value_distribution(collection, field))

        dominant_types = self.__dominant_range_types(stats.get('type_counts', {}))
        if dominant_types:
            stats.update(self.__fetch_field_range_stats(collection, field, dominant_types))
            histogram = self.__fetch_field_histogram(collection, field, dominant_types)
            if histogram:
                stats['histogram'] = histogram

        if stats.get('array_count', 0):
            stats['is_multikey'] = True

        return stats

    def __fetch_field_presence_stats(self, collection: Collection, field: str) -> dict:
        value_ref = f'${field}'
        summary = self.__aggregate_one(collection, [
            {'$project': {'_field_value': value_ref}},
            {'$project': {
                '_field_value': 1,
                '_is_array': {'$isArray': '$_field_value'},
                '_array_length': {
                    '$cond': [
                        {'$isArray': '$_field_value'},
                        {'$size': '$_field_value'},
                        0,
                    ],
                },
            }},
            {'$group': {
                '_id': None,
                'non_null_count': {'$sum': {'$cond': [{'$ne': ['$_field_value', None]}, 1, 0]}},
                'array_count': {'$sum': {'$cond': ['$_is_array', 1, 0]}},
                'array_total_length': {'$sum': {'$cond': ['$_is_array', '$_array_length', 0]}},
                'array_max_length': {'$max': {'$cond': ['$_is_array', '$_array_length', 0]}},
            }},
        ])

        array_count = int(summary.get('array_count', 0) or 0)
        output = {
            'non_null_count': int(summary.get('non_null_count', 0) or 0),
            'array_count': array_count,
            'array_avg_length': self.__ratio(float(summary.get('array_total_length', 0) or 0), float(array_count)),
            'array_max_length': int(summary.get('array_max_length', 0) or 0),
        }

        if array_count:
            array_summary = self.__aggregate_one(collection, [
                {'$project': {'_field_value': value_ref}},
                {'$match': {'$expr': {'$isArray': '$_field_value'}}},
                {'$project': {'_array_length': {'$size': '$_field_value'}}},
                {'$group': {
                    '_id': None,
                    'array_min_length': {'$min': '$_array_length'},
                    'array_avg_length': {'$avg': '$_array_length'},
                    'array_max_length': {'$max': '$_array_length'},
                }},
            ])
            output.update({
                'array_min_length': int(array_summary.get('array_min_length', 0) or 0),
                'array_avg_length': float(array_summary.get('array_avg_length', output['array_avg_length']) or 0.0),
                'array_max_length': int(array_summary.get('array_max_length', output['array_max_length']) or 0),
            })
        else:
            output['array_min_length'] = 0

        return output

    def __fetch_field_value_distribution(self, collection: Collection, field: str) -> dict:
        grouped = self.__aggregate_one(collection, self.__field_scalar_values_pipeline(field) + [
            {'$group': {'_id': '$value', 'count': {'$sum': 1}, 'type': {'$first': '$value_type'}}},
            {'$facet': {
                'top_values': [
                    {'$sort': {'count': -1}},
                    {'$limit': self.TOP_VALUE_LIMIT},
                    {'$project': {'_id': 0, 'value': '$_id', 'count': '$count', 'type': '$type'}},
                ],
                'distinct': [
                    {'$limit': self.DISTINCT_COUNT_CAP + 1},
                    {'$count': 'count'},
                ],
                'total_values': [
                    {'$group': {'_id': None, 'count': {'$sum': '$count'}}},
                ],
                'type_counts': [
                    {'$group': {'_id': '$type', 'count': {'$sum': '$count'}}},
                    {'$project': {'_id': 0, 'type': '$_id', 'count': '$count'}},
                ],
            }},
        ])

        distinct_count = self.__first_count(grouped.get('distinct'))
        value_count = self.__first_count(grouped.get('total_values'))
        type_counts = {
            item.get('type'): int(item.get('count', 0) or 0)
            for item in grouped.get('type_counts', [])
            if item.get('type')
        }

        return {
            'value_count': value_count,
            'distinct_count': distinct_count,
            'distinct_count_capped': distinct_count > self.DISTINCT_COUNT_CAP,
            'top_values': grouped.get('top_values', []),
            'type_counts': type_counts,
        }

    def __fetch_field_range_stats(self, collection: Collection, field: str, value_types: set[str]) -> dict:
        summary = self.__aggregate_one(collection, self.__field_scalar_values_pipeline(field) + [
            {'$match': {'value_type': {'$in': sorted(value_types)}}},
            {'$group': {
                '_id': None,
                'range_value_count': {'$sum': 1},
                'min': {'$min': '$value'},
                'max': {'$max': '$value'},
            }},
        ])

        return {
            'range_value_count': int(summary.get('range_value_count', 0) or 0),
            'min': summary.get('min'),
            'max': summary.get('max'),
        }

    def __fetch_field_histogram(self, collection: Collection, field: str, value_types: set[str]) -> list[dict]:
        buckets = self.__aggregate_list(collection, self.__field_scalar_values_pipeline(field) + [
            {'$match': {'value_type': {'$in': sorted(value_types)}}},
            {'$bucketAuto': {
                'groupBy': '$value',
                'buckets': self.HISTOGRAM_BUCKETS,
                'output': {'count': {'$sum': 1}},
            }},
        ])

        histogram = list[dict]()
        for bucket in buckets:
            bucket_id = bucket.get('_id', {})
            if not isinstance(bucket_id, dict):
                continue
            histogram.append({
                'min': bucket_id.get('min'),
                'max': bucket_id.get('max'),
                'count': int(bucket.get('count', 0) or 0),
            })
        return histogram

    def __field_scalar_values_pipeline(self, field: str) -> list[dict]:
        value_ref = f'${field}'
        return [
            {'$project': {'_field_value': value_ref}},
            {'$project': {
                '_values': {
                    '$cond': [
                        {'$isArray': '$_field_value'},
                        '$_field_value',
                        ['$_field_value'],
                    ],
                },
            }},
            {'$unwind': {'path': '$_values', 'preserveNullAndEmptyArrays': False}},
            {'$project': {'value': '$_values', 'value_type': {'$type': '$_values'}}},
            {'$match': {
                'value': {'$ne': None},
                'value_type': {'$in': sorted(self.SCALAR_VALUE_TYPES)},
            }},
        ]

    def __dominant_range_types(self, type_counts: dict) -> set[str]:
        numeric_count = sum(int(type_counts.get(type_name, 0) or 0) for type_name in self.NUMERIC_VALUE_TYPES)
        date_count = int(type_counts.get('date', 0) or 0)
        if numeric_count <= 0 and date_count <= 0:
            return set()
        if date_count > numeric_count:
            return {'date'}
        return self.NUMERIC_VALUE_TYPES

    def __aggregate_one(self, collection: Collection, pipeline: list[dict]) -> dict:
        rows = self.__aggregate_list(collection, pipeline)
        return rows[0] if rows else {}

    def __aggregate_list(self, collection: Collection, pipeline: list[dict]) -> list[dict]:
        try:
            return list(collection.aggregate(
                pipeline,
                allowDiskUse=True,
                maxTimeMS=self.FIELD_STATS_MAX_TIME_MS,
            ))
        except Exception:
            return []

    def __first_count(self, values) -> int:
        if not isinstance(values, list) or not values:
            return 0
        first = values[0]
        return int(first.get('count', 0) or 0) if isinstance(first, dict) else 0

    def __sample_type_name(self, value: Any) -> str:
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 'number'
        if isinstance(value, (datetime, date)):
            return 'date'
        if isinstance(value, str):
            return 'string'
        if value is None:
            return 'null'
        return type(value).__name__

    def __is_hashable_sample_value(self, value: Any) -> bool:
        if isinstance(value, (dict, list, set)):
            return False
        try:
            hash(value)
        except TypeError:
            return False
        return True

    def __ratio(self, numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0
