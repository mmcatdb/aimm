from abc import ABC, abstractmethod
from typing import Generic
from core.query import QueryInstance, QueryMeasurement, TQuery
from core.utils import ProgressTracker, plural, print_warning

class BasePlanExtractor(ABC, Generic[TQuery]):

    @abstractmethod
    def measure_query(self, query: TQuery) -> tuple[float, int]:
        """Measures single query execution. Returns tuple of (time, num_results)."""
        pass

    @abstractmethod
    def explain_query(self, query: TQuery, do_profile: bool) -> dict:
        """Gets query execution plan. Returns plan as dict.

        Args:
            do_profile: Whether to include execution statistics in the plan (e.g., PostgreSQL ANALYZE).
        """

        # The execution statistics are important for training the model, but they are not used when extracting features. So, we don't need them when evaluating the model.

        pass

    @abstractmethod
    def collect_global_stats(self) -> dict:
        """Collects global statistics about the database that are not specific to any single query, but might be relevant for feature extraction."""

        # Global stats might contain information about the kinds / database that are not present in the plans, but are still relevant for feature extraction.
        # They should be extracted together with the plans to ensure accurate values.

        pass

    def measure_and_explain_queries(self, queries: list[QueryInstance[TQuery]], num_runs: int) -> list[QueryMeasurement[TQuery]]:
        """Generates and measures queries.

        Args:
            num_queries: Total number of queries to generate and measure. Should be larger than the number of templates.
            num_runs: Number of executions per query for averaging
        """

        progress = ProgressTracker.limited(len(queries))
        progress.start(f'Measuring {plural(len(queries), "query", "queries")} ({num_runs} runs each) ... ')

        measurements = list[QueryMeasurement[TQuery]]()
        invalid_queries = list[QueryInstance[TQuery]]()

        for query in queries:
            is_exception = False
            try:
                measurement = self.measure_and_explain_query(query, num_runs)
                measurements.append(measurement)
            except Exception as e:
                is_exception = True
                print()
                # Don't print the stack trace as we don't care about some random internals of the driver.
                print_warning(f'Could not execute query {query.label}.', e, suppress_stacktrace=True)
                print()
                invalid_queries.append(query)
            progress.track(force_print=is_exception)

        progress.finish()
        print(f'\nCollected {plural(len(measurements), "measurement")} successfully.')

        if invalid_queries:
            queries_str = '\n'.join(f'  {q.label}' for q in invalid_queries)
            print()
            print_warning(f'Failed to execute {plural(len(invalid_queries), "query", "queries")}:\n{queries_str}')

        return measurements

    def measure_and_explain_query(self, query: QueryInstance[TQuery], num_runs: int) -> QueryMeasurement[TQuery]:
        """Returns a comprehensive measurement of the query, including the execution plan and latency.

        The query is run `num_runs` times to get a more stable latency measurement.
        After that, the plan is extracted, ensuring that the best possible plan is captured.
        """
        times = list[float]()
        for _ in range(num_runs):
            time, _ = self.measure_query(query.content)
            times.append(time)

        plan = self.explain_query(query.content, do_profile=True)

        return QueryMeasurement.from_instance(query, plan, times)
