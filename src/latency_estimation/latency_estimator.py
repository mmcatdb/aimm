from typing import Generic
from core.query import  DatabaseId, DriverType, QueryInstance, QueryInstanceId, QueryMeasurement, TQuery, parse_database_id, parse_query_instance_id
from .feature_extractor import BaseFeatureExtractor
from .plan_extractor import BasePlanExtractor
from .model import BaseModel

class LatencyEstimator(Generic[TQuery]):
    """Estimates query latency using a trained model without executing queries.

    Uses EXPLAIN to get the query plan and neural network for estimation.
    """

    def __init__(self):
        self.__per_driver_type = dict[DriverType, tuple[BaseFeatureExtractor, BaseModel]]()
        """Contains tuples of (feature_extractor, model) for each registered driver type."""
        self.__per_database = dict[DatabaseId, tuple[BasePlanExtractor, dict, DriverType]]()
        """Contains tuples of (plan_extractor, global_stats, driver_type) for each registered database. Used for caching the global stats to avoid redundant collection."""

        self.__estimate_cache = dict[QueryInstanceId, float]()
        """Cache for estimated latencies of queries."""
        self.__measure_cache = dict[QueryInstanceId, QueryMeasurement]()
        """Cache for measured latencies of queries."""

    def register_driver_type(self, driver_type: DriverType, feature_extractor: BaseFeatureExtractor, model: BaseModel):
        """Each driver type has to be registered so that the corresponding feature extractor and model can be used for estimation."""
        self.__per_driver_type[driver_type] = (feature_extractor, model)

    def register_database(self, database_id: DatabaseId, plan_extractor: BasePlanExtractor):
        """Each database has to be registered so that the plans can be extracted for the queries."""
        stats = plan_extractor.collect_global_stats()
        driver_type, _, _ = parse_database_id(database_id)
        self.__per_database[database_id] = (plan_extractor, stats, driver_type)

    def estimate(self, query: QueryInstance) -> float:
        """Estimates query latency without executing the query. Returns the latency in milliseconds. The result is cached by the query_id."""
        value = self.__estimate_cache.get(query.id)
        if value is  None:
            database_id, _, _ = parse_query_instance_id(query.id)
            value = self.estimate_uncached(database_id, query.content)
            self.__estimate_cache[query.id] = value

        return value

    def estimate_uncached(self, database_id: DatabaseId, content: TQuery) -> float:
        """Estimates query latency without executing the query. Returns the latency in milliseconds."""
        plan_extractor, global_stats, driver_type = self.__per_database[database_id]
        feature_extractor, model = self.__per_driver_type[driver_type]

        raw_plan = plan_extractor.explain_query(content, do_profile=False)
        feature_extractor.set_global_stats(global_stats)
        plan = feature_extractor.extract_plan(raw_plan)

        return model.evaluate(plan)

    def measure(self, query: QueryInstance, num_runs: int) -> QueryMeasurement:
        """Measures the actual latency of a query by executing it multiple times. Returns a list of latencies in milliseconds."""
        value = self.__measure_cache.get(query.id)
        if value is None:
            value = self.measure_uncached(query, num_runs)
            self.__measure_cache[query.id] = value

        return value

    def measure_uncached(self, query: QueryInstance, num_runs: int) -> QueryMeasurement:
        """Measures the actual latency of a query by executing it multiple times. Returns a list of latencies in milliseconds."""
        database_id, _, _ = parse_query_instance_id(query.id)
        plan_extractor, _, _ = self.__per_database[database_id]

        return plan_extractor.measure_and_explain_query(query.content, num_runs=num_runs)
