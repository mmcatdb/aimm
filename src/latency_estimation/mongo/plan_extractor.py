import time
import numpy as np
from common.database import Database, MongoQuery, MongoFindQuery, MongoAggregateQuery
from common.drivers import MongoDriver
from latency_estimation.mongo.trainer import MongoDataset
from common.utils import ProgressTracker

class PlanExtractor:
    """Extracts query plans and execution statistics from MongoDB."""

    def __init__(self, driver: MongoDriver, database: Database[MongoQuery]):
        self.config = driver
        self.db = driver.database()
        self.database = database

    def collect_training_dataset(self, num_queries: int, num_runs: int) -> MongoDataset:
        """
        Collect training dataset: explain plans + actual execution times.
        For each query, we collect:
        - explain('executionStats') for the plan tree + per-stage stats
        - Wall-clock execution time averaged over num_runs
        """
        queries = self.database.get_train_queries(num_queries)

        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Collecting {len(queries)} query plans ({num_runs} runs each) ... ')

        plans = []
        execution_times = []

        for i, query in enumerate(queries):
            try:
                if isinstance(query, MongoFindQuery):
                    plan = self.explain_find(query, verbosity='executionStats')
                    times = self.measure_find(query, num_runs=num_runs)
                else:
                    plan = self.explain_aggregate(query, verbosity='executionStats')
                    times = self.measure_aggregate(query, num_runs=num_runs)

                plans.append(plan)
                execution_times.append(np.mean(times))
                progress.track()

            except Exception as e:
                print(f'\nError executing query {i}: {e}')
                # print(f'Query preview: {query[:100]}...\n')
                continue

        progress.finish()

        dataset = MongoDataset(queries, plans, execution_times)

        print(f'\nCollected {len(dataset)} samples successfully')
        return dataset

    # ------------------------------------------------------------------
    # Explain helpers
    # ------------------------------------------------------------------

    def explain_find(self, query: MongoFindQuery, verbosity: str) -> dict:
        """Run explain on a find command."""
        cmd: dict = {
            'find': query.collection,
            'filter': query.filter,
        }

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

    def measure_find(self, query: MongoFindQuery, num_runs: int) -> list[float]:
        """
        Execute a find query and measure wall-clock time.
        Returns (mean_ms, min_ms, max_ms).
        """
        collection = self.db[query.collection]
        times = []

        for _ in range(num_runs):
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
            list(cursor)  # force materialization
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        return times

    def explain_aggregate(self, query: MongoAggregateQuery, verbosity: str) -> dict:
        """Run explain on an aggregate pipeline."""
        cmd: dict = {
            'aggregate': query.collection,
            'pipeline': query.pipeline,
            'cursor': {}
        }

        explain = self.db.command('explain', cmd, verbosity=verbosity)
        return self.extract_plan(explain)

    def measure_aggregate(self, query: MongoAggregateQuery, num_runs: int) -> list[float]:
        """Execute an aggregate pipeline and measure wall-clock time."""
        collection = self.db[query.collection]
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            list(collection.aggregate(query.pipeline))
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        return times

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
