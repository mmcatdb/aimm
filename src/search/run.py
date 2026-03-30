import json
import sys
from search.mcts import MCTS, StateMapping, StateMatrix
from common.config import DatasetName
from common.drivers import DriverType
from common.database import MongoQuery
from latency_estimation.postgres.context import PostgresContext
from latency_estimation.postgres.latency_estimator import LatencyEstimator as PostgresLatencyEstimator
from latency_estimation.mongo.context import MongoContext
from latency_estimation.mongo.latency_estimator import LatencyEstimator as MongoLatencyEstimator
from latency_estimation.neo4j.context import Neo4jContext
from latency_estimation.neo4j.latency_estimator import LatencyEstimator as Neo4jLatencyEstimator

QueriesForKind = dict[str, list]
"""Mapping of query kind (e.g., 'test:1') to list of query objects (e.g., list of MongoQuery)."""

# TODO unify with common.config
DATASET = DatasetName.EDBT
MAX_ITERATIONS = 1000000
IS_QUIET = True
SCALE = 1.0

def main():
    from datasets.edbt.postgres_database import EdbtPostgresDatabase
    from datasets.edbt.mongo_database import EdbtMongoDatabase
    from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
    import ast

    query_weights_str = "".join(sys.argv[1:])
    query_weights = ast.literal_eval(query_weights_str)

    postgres_estimator = load_postgres_estimator("data/checkpoints/edbt_postgres_best.pt")
    mongo_estimator = load_mongo_estimator("data/checkpoints/edbt_mongo_best.pt")
    neo4j_estimator = load_neo4j_estimator("data/checkpoints/edbt_neo4j_best.pt")

    postgres_db = EdbtPostgresDatabase(SCALE)
    mongo_db = EdbtMongoDatabase(SCALE)
    neo4j_db = EdbtNeo4jDatabase(SCALE)

    postgres_queries = {q.id: q.generate() for q in postgres_db.get_query_defs()}
    mongo_queries = {q.id: q.generate() for q in mongo_db.get_query_defs()}
    neo4j_queries = {q.id: q.generate() for q in neo4j_db.get_query_defs()}

    qtt_postgres = extract_postgres_tables(postgres_queries)
    qtt_mongo = extract_mongo_tables(mongo_queries)
    qtt_neo = extract_neo4j_tables(neo4j_queries)

    query_engine = QueryEngine(
        postgres_estimator,
        postgres_queries,
        mongo_estimator,
        mongo_queries,
        neo4j_estimator,
        neo4j_queries,
        query_weights
    )
    output_collector = OutputCollector(qtt_postgres, qtt_mongo, qtt_neo)

    dbs = [DriverType.POSTGRES.value, DriverType.MONGO.value, DriverType.NEO4J.value]

    NUM_QUERIES = 11
    kinds = list(range(NUM_QUERIES))
    initial_mapping = [DriverType.POSTGRES.value] * len(kinds)
    mcts = MCTS(query_engine, kinds, output_collector, dbs)

    # Run MCTS to find the best execution plan
    mcts.run(initial_mapping, MAX_ITERATIONS)

def load_postgres_estimator(checkpoint_path: str | None) -> PostgresLatencyEstimator:
    ctx = PostgresContext.create(SCALE, quiet=IS_QUIET, dataset=DATASET)
    model = ctx.load_model(checkpoint_path)
    return PostgresLatencyEstimator(ctx.extractor, model)

def load_mongo_estimator(checkpoint_path: str | None) -> MongoLatencyEstimator:
    ctx = MongoContext.create(SCALE, quiet=IS_QUIET, dataset=DATASET)
    model = ctx.load_model(checkpoint_path)
    return MongoLatencyEstimator(ctx.extractor, model)

def load_neo4j_estimator(checkpoint_path: str | None) -> Neo4jLatencyEstimator:
    ctx = Neo4jContext.create(SCALE, quiet=IS_QUIET, dataset=DATASET)
    model = ctx.load_model(checkpoint_path)
    return Neo4jLatencyEstimator(ctx.extractor, model)

class QueryEngine:
    def __init__(self,
        postgres_estimator,
        postgres_queries: dict[str, str],
        mongo_estimator,
        mongo_queries: dict[str, MongoQuery],
        neo4j_estimator,
        neo4j_queries: dict[str, str],
        query_weights: list[float]
    ):
        self.postgres_estimator = postgres_estimator
        self.postgres_queries = postgres_queries
        self.mongo_estimator = mongo_estimator
        self.mongo_queries = mongo_queries
        self.neo4j_estimator = neo4j_estimator
        self.neo4j_queries = neo4j_queries
        self.query_weights = query_weights

    def estimate_latency(self, state: StateMapping) -> float:
        """
        Estimate total latency for query-assignment state.

        State semantics:
        - Rows are databases, columns are queries.
        - Exactly one True per query column indicates selected execution DB.
        """
        total_latency = 0.0

        for query_index, weight in enumerate(self.query_weights):
            type = DriverType(state[query_index])
            query_id = f"test:{query_index}"

            if type == DriverType.POSTGRES:
                query = self.postgres_queries.get(query_id)
                estimator = self.postgres_estimator
            elif type == DriverType.MONGO:
                query = self.mongo_queries.get(query_id)
                estimator = self.mongo_estimator
            elif type == DriverType.NEO4J:
                query = self.neo4j_queries.get(query_id)
                estimator = self.neo4j_estimator
            else:
                raise ValueError(f"Unsupported database: {type.value}")

            if not query:
                raise ValueError(f"No queries found for query_id={query_id} in database={type.value}")

            latency, _ = estimator.estimate(query)
            if type == DriverType.NEO4J:
                latency /= 30

            total_latency += float(latency) * weight

        return total_latency

AdaptationSchema = dict[str, list[str]]
"""Map of database name -> list of assigned query kinds."""

class AdaptationSolution:
    def __init__(self, iteration: int, price: float, objexes: AdaptationSchema):
        self.iteration = iteration
        self.price = price
        self.objexes = objexes

    def to_dict(self) -> dict:
        return {
            'id': self.iteration,
            'price': self.price,
            'objexes': self.objexes,
        }

QttMapping = dict[str, set[str]]
"""qtt means query-to-tables mapping for each database, i.e. dict[query_id, set of tables used in that query]"""

def extract_postgres_tables(queries: dict[str, str]) -> QttMapping:
    from latency_estimation.postgres.feature_extractor import FeatureExtractor as PostgresFeatureExtractor
    extractor = PostgresFeatureExtractor()
    return {query_id: extractor.extract_query_kinds(query) for query_id, query in queries.items()}

def extract_mongo_tables(queries: dict[str, MongoQuery]) -> QttMapping:
    from latency_estimation.mongo.feature_extractor import FeatureExtractor as MongoFeatureExtractor
    extractor = MongoFeatureExtractor()
    return {query_id: extractor.extract_query_kinds(query) for query_id, query in queries.items()}

def extract_neo4j_tables(queries: dict[str, str]) -> QttMapping:
    from latency_estimation.neo4j.feature_extractor import FeatureExtractor as Neo4jFeatureExtractor
    extractor = Neo4jFeatureExtractor()
    return {query_id: extractor.extract_query_kinds(query) for query_id, query in queries.items()}

class OutputCollector:
    def __init__(self, qtt_postgres: QttMapping, qtt_mongo: QttMapping, qtt_neo4j: QttMapping):
        self.qtt_postgres = qtt_postgres
        self.qtt_mongo = qtt_mongo
        self.qtt_neo4j = qtt_neo4j
        self.outputs = list[AdaptationSolution]()

    def on_initial_solution(self, state: StateMapping, latency: float):
        initial_schema = self.__convert_state_to_schema(state)
        self.outputs.append(AdaptationSolution(0, latency, initial_schema))
        self.output_best_schemas()

    def on_best_solution(self, state: StateMapping, latency: float, processed_states: int):
        schema = self.__convert_state_to_schema(state)
        self.outputs.append(AdaptationSolution(processed_states, latency, schema))
        self.output_best_schemas()

    def __convert_state_to_schema(self, state: StateMapping) -> AdaptationSchema:
        schema = dict[str, set[str]]()
        for query_index, db in enumerate(state):
            if db not in schema:
                schema[db] = set()

            query_id = f"test:{query_index}"

            if db == DriverType.POSTGRES.value:
                kinds = self.qtt_postgres.get(query_id, [])
            elif db == DriverType.MONGO.value:
                kinds = self.qtt_mongo.get(query_id, [])
            elif db == DriverType.NEO4J.value:
                kinds = self.qtt_neo4j.get(query_id, [])
            else:
                raise ValueError(f"Unsupported database: {db}")

            schema[db].update(kinds)

        return {db: list(kinds) for db, kinds in schema.items()}

    def output_best_schemas(self):
        output = {
            'processedStates': self.outputs[-1].iteration,
            'solutions': [solution.to_dict() for solution in self.outputs[-3:]],
        }
        json_string = json.dumps(output)
        sys.stdout.write(json_string + '\n')
        sys.stdout.flush()

    def on_iteration(self, iteration: int, processed_states: int):
        pass

if __name__ == "__main__":
    main()
