import torch
from latency_estimation.neo4j.plan_extractor import PlanExtractor
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork

# TODO: It's the same as postgres - unify

class LatencyEstimator:
    """
    Estimates query latency using a trained model without executing queries.
    Uses EXPLAIN to get the query plan and neural network for estimation.
    """
    def __init__(self, model: PlanStructuredNetwork, extractor: PlanExtractor):
        self.model = model
        self.extractor = extractor

    def estimate_batch(self, queries: list[str]) -> list[tuple[str, float, dict]]:
        """
        Estimate latency for multiple queries.
        Args:
            queries: List of Cypher query strings
        Returns:
            List of tuples (query, estimated_latency_seconds, plan)
        """
        results = []
        for query in queries:
            try:
                latency, plan = self.estimate(query)
                results.append((query, latency, plan))
            except Exception as e:
                results.append((query, None, {'error': str(e)}))

        return results

    def estimate(self, query: str) -> tuple[float, dict]:
        """
        Estimate query latency without executing the query.
        Args:
            query: Cypher query string
        Returns:
            Tuple of (estimated_latency_seconds, query_plan)
        """
        plan = self.extractor.explain_plan(query)

        # Estimate latency using the trained model
        with torch.no_grad():
            estimated_latency = self.model.estimate_plan_latency(plan)

        return estimated_latency.item(), plan
