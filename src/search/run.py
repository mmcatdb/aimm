from search.mcts import MCTS
from common.config import DatasetName
from common.utils import auto_close
from search.queries import get_postgres_queries, get_neo4j_queries
from latency_estimation.postgres.context import PostgresContext
from latency_estimation.postgres.latency_estimator import LatencyEstimator as PostgresLatencyEstimator
from latency_estimation.neo4j.context import Neo4jContext
from latency_estimation.neo4j.latency_estimator import LatencyEstimator as Neo4jLatencyEstimator
from latency_estimation.mongo.context import MongoContext
from latency_estimation.mongo.latency_estimator import LatencyEstimator as MongoLatencyEstimator
from common.database import MongoQuery, MongoAggregateQuery, MongoFindQuery, try_parse_mongo_query
import re


def extract_neo4j_tables(queries: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """
    Takes a dict of Neo4j queries keyed by query id and returns a dict where
    each value is a tuple of node labels (tables) used in that query.
    """
    results = {}
    # Regex to match Neo4j node labels, e.g., (alias:Label) or (:Label)
    pattern = re.compile(r'\(\s*\w*\s*:\s*([A-Za-z0-9_]+)')
    
    for query_id, query in queries.items():
        labels = pattern.findall(query)
        # Use dict.fromkeys to remove duplicates while preserving order
        unique_labels = tuple(dict.fromkeys(labels))
        results[query_id] = unique_labels
        
    return results


def extract_postgres_tables(queries: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """
    Takes a dict of Postgres queries keyed by query id and returns a dict where
    each value is a tuple of table names used in the corresponding query.

    It captures sources after FROM/JOIN (including quoted names like "order"),
    and excludes CTE aliases defined in a WITH clause.
    """
    results = {}

    cte_pattern = re.compile(r'\bWITH\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', re.IGNORECASE)
    cte_following_pattern = re.compile(r',\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', re.IGNORECASE)

    source_pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:(?:"([^"]+)")|([A-Za-z_][A-Za-z0-9_]*))(?:\s*\.\s*(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*)))?',
        re.IGNORECASE,
    )

    for query_id, query in queries.items():
        ctes = {
            *cte_pattern.findall(query),
            *cte_following_pattern.findall(query),
        }
        ctes_lower = {cte.lower() for cte in ctes}

        tables = []
        for match in source_pattern.finditer(query):
            first_quoted, first_plain, second_quoted, second_plain = match.groups()

            first_part = first_quoted or first_plain
            second_part = second_quoted or second_plain
            table_name = second_part if second_part else first_part

            if not table_name:
                continue
            if table_name.lower() in ctes_lower:
                continue
            tables.append(table_name)

        unique_tables = tuple(dict.fromkeys(tables))
        results[query_id] = unique_tables

    return results


def _extract_mongo_pipeline_collections(pipeline: list[dict]) -> tuple[str, ...]:
    collections: list[str] = []

    def append_collection(name: str | None):
        if isinstance(name, str) and name:
            collections.append(name)

    def visit_node(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('$lookup', '$graphLookup') and isinstance(value, dict):
                    append_collection(value.get('from'))
                    if 'pipeline' in value:
                        visit_node(value['pipeline'])
                    continue

                if key == '$unionWith':
                    if isinstance(value, str):
                        append_collection(value)
                    elif isinstance(value, dict):
                        append_collection(value.get('coll'))
                        if 'pipeline' in value:
                            visit_node(value['pipeline'])
                    continue

                visit_node(value)
        elif isinstance(node, list):
            for item in node:
                visit_node(item)

    visit_node(pipeline)
    return tuple(dict.fromkeys(collections))


def extract_mongo_tables(queries: dict[str, MongoQuery | str]) -> dict[str, tuple[str, ...]]:
    """
    Takes a dict of Mongo queries keyed by query id and returns a dict where
    each value is a tuple of collection names required by that query.

    Includes the main collection and additional collections referenced in
    aggregate pipeline stages such as $lookup, $graphLookup, and $unionWith.
    """
    results = {}

    for query_id, query in queries.items():
        mongo_query = query if isinstance(query, (MongoFindQuery, MongoAggregateQuery)) else try_parse_mongo_query(query)
        if mongo_query is None:
            results[query_id] = tuple()
            continue

        collections = [mongo_query.collection]
        if isinstance(mongo_query, MongoAggregateQuery):
            collections.extend(_extract_mongo_pipeline_collections(mongo_query.pipeline))

        results[query_id] = tuple(dict.fromkeys(collections))

    return results



def load_postgres_estimator(checkpoint_path: str | None = None, dataset: DatasetName = DatasetName.EDBT, quiet: bool = True) -> tuple[PostgresContext, PostgresLatencyEstimator]:
    """Load Postgres latency estimator and return (context, estimator).

    The caller is responsible for closing the returned context.
    """
    ctx = PostgresContext.create(quiet=quiet, dataset=dataset)
    model = ctx.load_model(checkpoint_path)
    estimator = PostgresLatencyEstimator(ctx.extractor, model)
    return ctx, estimator

def load_neo4j_estimator(
    checkpoint_path: str | None = None, dataset: DatasetName = DatasetName.EDBT, quiet: bool = True) -> tuple[Neo4jContext, Neo4jLatencyEstimator]:
    """Load Neo4j latency estimator and return (context, estimator).

    The caller is responsible for closing the returned context.
    """
    ctx = Neo4jContext.create(quiet=quiet, dataset=dataset)
    model = ctx.load_model(checkpoint_path)
    estimator = Neo4jLatencyEstimator(ctx.extractor, model)
    return ctx, estimator

def load_mongo_estimator(
    checkpoint_path: str | None = None, dataset: DatasetName = DatasetName.EDBT, quiet: bool = True) -> tuple[MongoContext, MongoLatencyEstimator]:
    """Load Mongo latency estimator and return (context, estimator).

    The caller is responsible for closing the returned context.
    """
    ctx = MongoContext.create(quiet=quiet, dataset=dataset)
    model = ctx.load_model(checkpoint_path)
    estimator = MongoLatencyEstimator(ctx.extractor, model)
    return ctx, estimator

class QueryEngine:
    def __init__(self, postgres_estimator, neo4j_estimator, postgres_queries, neo4j_queries, mongo_estimator, mongo_queries):
        self.postgres_estimator = postgres_estimator
        self.neo4j_estimator = neo4j_estimator
        self.postgres_queries = postgres_queries
        self.neo4j_queries = neo4j_queries
        self.mongo_estimator = mongo_estimator
        self.mongo_queries = mongo_queries

    def _normalize_database_name(self, database: str) -> str:
        lowered = database.strip().lower()
        if lowered in {"postgres", "postgresql"}:
            return "postgres"
        if lowered in {"neo4j"}:
            return "neo4j"
        if lowered in {"mongo", "mongodb"}:
            return "mongo"
        raise ValueError(f"Unsupported database: {database}")

    def estimate_latency(self, state, mcts):
        """
        Estimate total latency for query-assignment state.

        State semantics:
        - Rows are databases, columns are queries.
        - Exactly one True per query column indicates selected execution DB.
        """
        query_to_database = mcts.state_to_mapping(state)
        total_latency = 0.0

        for query_id in mcts.query_ids:
            database = query_to_database[query_id]
            normalized_database = self._normalize_database_name(database)

            if normalized_database == "postgres":
                queries = self.postgres_queries.get(f'test:{query_id}')
                estimator = self.postgres_estimator
            elif normalized_database == "neo4j":
                queries = self.neo4j_queries.get(f'test:{query_id}')
                estimator = self.neo4j_estimator
            elif normalized_database == "mongo":
                queries = self.mongo_queries.get(f'test:{query_id}')
                estimator = self.mongo_estimator
            else:
                raise ValueError(f"Unsupported database: {database}")

            if not queries:
                raise ValueError(f"No queries found for query_id={query_id} in database={database}")

            for query in queries:
                latency, _ = estimator.estimate(query)
                if normalized_database == "neo4j":
                    latency /= 30
                total_latency += float(latency)

        return total_latency
    
    
class SchemaConverter:
    def __init__(self, query_to_tables_postgres, query_to_tables_neo4j, query_to_tables_mongo):
        self.query_to_tables_postgres = query_to_tables_postgres
        self.query_to_tables_neo4j = query_to_tables_neo4j
        self.query_to_tables_mongo = query_to_tables_mongo
        
    def convert_state_to_schema(self, state):
        schema = {}
        for query_id, db in state.items():
            if db not in schema:
                schema[db] = set()
            
            query_key = f"test:{query_id}"
            db_lower = db.lower()
            
            if "postgres" in db_lower:
                tables = self.query_to_tables_postgres.get(query_key, [])
            elif "neo4j" in db_lower:
                tables = self.query_to_tables_neo4j.get(query_key, [])
            elif "mongo" in db_lower:
                tables = self.query_to_tables_mongo.get(query_key, [])
            else:
                tables = []
                
            schema[db].update(tables)
            
        return {db: list(tables) for db, tables in schema.items()}
    

    
    
if __name__ == "__main__":
    from datasets.edbt.postgres_database import EdbtPostgresDatabase
    from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
    from datasets.edbt.mongo_database import EdbtMongoDatabase

    postgres_ctx, postgres_estimator = load_postgres_estimator("data/checkpoints/edbt_postgres_best.pt")
    neo4j_ctx, neo4j_estimator = load_neo4j_estimator("data/checkpoints/edbt_neo4j_best.pt")
    mongo_ctx, mongo_estimator = load_mongo_estimator("data/checkpoints/edbt_mongo_best.pt")

    postgres_db = EdbtPostgresDatabase()
    neo4j_db = EdbtNeo4jDatabase()
    mongo_db = EdbtMongoDatabase()

    postgres_queries = {q.id: [q.generate()] for q in postgres_db.get_query_defs()}
    neo4j_queries = {q.id: [q.generate()] for q in neo4j_db.get_query_defs()}
    mongo_queries = {q.id: [q.generate()] for q in mongo_db.get_query_defs()}
    
    

    neo_queries = {query_id: q[0] for query_id, q in neo4j_queries.items()}
    query_to_tables_neo = extract_neo4j_tables(neo_queries)
    
    pg_queries = {query_id: q[0] for query_id, q in postgres_queries.items()}
    query_to_tables_postgres = extract_postgres_tables(pg_queries)

    mongo_query_map = {query_id: q[0] for query_id, q in mongo_queries.items()}
    query_to_tables_mongo = extract_mongo_tables(mongo_query_map)
    
    query_engine = QueryEngine(postgres_estimator, neo4j_estimator, postgres_queries, neo4j_queries, mongo_estimator, mongo_queries)
    schema_converter = SchemaConverter(query_to_tables_postgres, query_to_tables_neo, query_to_tables_mongo)
    
    
    dbs = ["Postgres", "Neo4j", "Mongodb"]
    
    NUM_QUERIES = 11
    kinds = list(range(NUM_QUERIES))
    initial_mapping = ["Postgres"]*len(kinds)
    mcts = MCTS(query_engine, kinds, schema_converter,dbs)
    
    # Run MCTS to find the best execution plan
    best_plan = mcts.run(initial_mapping)
    
    # print("Best execution plan:", best_plan)
    
