from core.driver_provider import Driver
from core.drivers import DriverType, MongoDriver, Neo4jDriver, PostgresDriver
from .latency_estimator import LatencyEstimator
from .model_evaluator import BaseModelEvaluator
from .feature_extractor import BaseFeatureExtractor
from .plan_extractor import BasePlanExtractor

def get_plan_extractor(driver: Driver) -> BasePlanExtractor:
    if isinstance(driver, PostgresDriver):
        from .postgres.plan_extractor import PlanExtractor
        return PlanExtractor(driver)
    elif isinstance(driver, MongoDriver):
        from .mongo.plan_extractor import PlanExtractor
        return PlanExtractor(driver)
    elif isinstance(driver, Neo4jDriver):
        from .neo4j.plan_extractor import PlanExtractor
        return PlanExtractor(driver)

def get_feature_extractor(driver_type: DriverType) -> BaseFeatureExtractor:
    if driver_type == DriverType.POSTGRES:
        from .postgres.feature_extractor import FeatureExtractor
        return FeatureExtractor()
    elif driver_type == DriverType.MONGO:
        from .mongo.feature_extractor import FeatureExtractor
        return FeatureExtractor()
    elif driver_type == DriverType.NEO4J:
        from .neo4j.feature_extractor import FeatureExtractor
        return FeatureExtractor()

def get_model_evaluator(driver_type: DriverType, estimator: LatencyEstimator) -> BaseModelEvaluator:
    if driver_type == DriverType.POSTGRES:
        from .postgres.model_evaluator import ModelEvaluator
        return ModelEvaluator(estimator)
    elif driver_type == DriverType.MONGO:
        from .mongo.model_evaluator import ModelEvaluator
        return ModelEvaluator(estimator)
    elif driver_type == DriverType.NEO4J:
        from .neo4j.model_evaluator import ModelEvaluator
        return ModelEvaluator(estimator)
