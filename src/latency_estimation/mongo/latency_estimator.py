import torch
from common.database import MongoAggregateQuery, MongoFindQuery, MongoQuery
from latency_estimation.mongo.plan_extractor import PlanExtractor
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork

class LatencyEstimator:
    """
    Estimates query latency using a trained model without executing queries.
    Uses EXPLAIN to get the query plan and neural network for estimation.
    """

    def __init__(self, extractor: PlanExtractor, model: PlanStructuredNetwork):
        self.extractor = extractor
        self.model = model

    def estimate_batch(self, queries: list[MongoQuery]) -> list[tuple[MongoQuery, float | None, dict]]:
        """
        Estimate latency for multiple queries.
        Args:
            queries: List of parsed Mongo query objects
        Returns:
            List of tuples (query, estimated_latency_ms, plan)
        """
        results = list[tuple[MongoQuery, float | None, dict]]()
        for query in queries:
            try:
                latency, plan = self.estimate(query)
                results.append((query, latency, plan))
            except Exception as e:
                results.append((query, None, {'error': str(e)}))

        return results

    def estimate(self, query: MongoQuery) -> tuple[float, dict]:
        """
        Estimate query latency without executing the query.
        Args:
            query: Parsed Mongo query object
        Returns:
            Tuple of (estimated_latency_ms, query_plan)
        """
        plan = self.extractor.explain_query(query, verbosity='queryPlanner')

        with torch.no_grad():
            estimated_latency = self.model(plan, query.collection)

        return estimated_latency.item(), plan
