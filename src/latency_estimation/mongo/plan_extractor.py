import time
from typing_extensions import override
from common.database import MongoQuery, MongoFindQuery, MongoAggregateQuery
from common.drivers import MongoDriver
from common.utils import ProgressTracker, print_warning, time_quantity
from common.query_registry import QueryDefMap
from latency_estimation.common import ArrayDataset
from latency_estimation.feature_extractor import BaseDatasetItem
from latency_estimation.plan_extractor import BasePlanExtractor

class MongoItem(BaseDatasetItem):
    def __init__(self, id: str, query: MongoQuery, plan: dict, times: list[float]):
        super().__init__(id, plan, times)
        self.query = query

    @override
    def query_string(self) -> str:
        return str(self.query)

class PlanExtractor(BasePlanExtractor[MongoQuery]):
    """Extracts query plans and execution statistics from MongoDB."""

    def __init__(self, driver: MongoDriver):
        self.config = driver
        self.db = driver.database()

    def create_dataset(self, queries: list[MongoQuery], num_runs: int, def_map: QueryDefMap[MongoQuery]) -> ArrayDataset[MongoItem]:
        """
        Collect training dataset: explain plans + actual execution times.
        For each query, we collect:
        - explain('executionStats') for the plan tree + per-stage stats
        - Wall-clock execution time averaged over num_runs
        """
        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        items = list[MongoItem]()

        for i, query in enumerate(queries):
            try:
                plan = self.explain_query(query, verbosity='executionStats')
                times = self.measure_query_multiple(query, num_runs)
                items.append(MongoItem(def_map[id(query)].id, query, plan, times))
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

        print(f'\nCollected {len(dataset)} samples successfully')
        return dataset

    # ------------------------------------------------------------------
    # Explain helpers
    # ------------------------------------------------------------------

    def explain_query(self, query: MongoQuery, verbosity: str) -> dict:
        if isinstance(query, MongoFindQuery):
            return self.__explain_find(query, verbosity)
        else:
            return self.__explain_aggregate(query, verbosity)

    def __explain_find(self, query: MongoFindQuery, verbosity: str) -> dict:
        cmd: dict = {
            'find': query.collection,
        }

        if query.filter:
            cmd['filter'] = query.filter
        if query.projection:
            cmd['projection'] = query.projection
        if query.sort:
            cmd['sort'] = query.sort
        if query.limit:
            cmd['limit'] = query.limit
        if query.skip:
            cmd['skip'] = query.skip

        explain = self.db.command('explain', cmd, verbosity=verbosity)
        return self.extract_plan(explain)

    def __explain_aggregate(self, query: MongoAggregateQuery, verbosity: str) -> dict:
        cmd: dict = {
            'aggregate': query.collection,
            'pipeline': query.pipeline,
            'cursor': {}
        }

        explain = self.db.command('explain', cmd, verbosity=verbosity)
        return self.extract_plan(explain)

    @override
    def measure_query(self, query: MongoQuery) -> tuple[float, int]:
        if isinstance(query, MongoFindQuery):
            return self.__measure_find(query)
        else:
            return self.__measure_aggregate(query)

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
        elapsed_ms = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed_ms, len(results)

    def __measure_aggregate(self, query: MongoAggregateQuery) -> tuple[float, int]:
        collection = self.db[query.collection]

        start = time.perf_counter()
        results = list(collection.aggregate(query.pipeline))
        elapsed_ms = time_quantity.to_base(time.perf_counter() - start, 's')

        return elapsed_ms, len(results)

    @staticmethod
    def extract_plan(explain_output: dict) -> dict:
        """
        Extract the plan tree from an explain output.
        Handles both explainVersion 1 (classic) and 2 (SBE / aggregation).
        """
        version = explain_output.get('explainVersion', '1')

        if version == '2':
            # SBE or aggregate: may have stages[] or queryPlanner.winningPlan.queryPlan
            if 'stages' in explain_output:
                # Aggregate pipeline: get the cursor's queryPlan
                cursor_stage = explain_output['stages'][0]
                if '$cursor' in cursor_stage:
                    wp = cursor_stage['$cursor']['queryPlanner']['winningPlan']
                    return wp.get('queryPlan', wp)
            wp = explain_output.get('queryPlanner', {}).get('winningPlan', {})
            return wp.get('queryPlan', wp)
        else:
            # Classic engine
            wp = explain_output.get('queryPlanner', {}).get('winningPlan', {})
            return wp
