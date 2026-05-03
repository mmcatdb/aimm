import json
import sys
import ast
from core.dynamic_provider import get_dynamic_class_instance
from core.query import QueryInstance, QueryInstanceId, QueryRegistry, SchemaId, create_database_id, create_query_instance_id, parse_schema_id
from latency_estimation.class_provider import get_plan_extractor
from latency_estimation.dataset import create_dataset_id
from latency_estimation.feature_extractor import load_feature_extractor
from latency_estimation.latency_estimator import LatencyEstimator
from latency_estimation.model import create_checkpoint_id, create_model_id
from providers.contex import Context
from .kind_extractor import get_kind_extractor
from .mcts import MCTS, StateMapping
from core.drivers import DriverType

# TODO unify with common.config
MAX_ITERATIONS = 1000000
IS_QUIET = True

# FIXME
SCHEMA_ID = 'edbt-1'
DATASET_NAME = 'mcts'
MODEL_NAME = 'mcts'
CHECKPOINT_NAME = 'best'

def main():
    query_weights_str = "".join(sys.argv[1:])
    query_weights = ast.literal_eval(query_weights_str)

    ctx = Context.default(quiet=IS_QUIET)

    estimator = LatencyEstimator()
    for driver_type in DriverType:
        estimator.register_driver_type(*load_driver_type(ctx, driver_type))
        estimator.register_database_extractor(*load_database(ctx, driver_type, SCHEMA_ID))

    queires = dict[QueryInstanceId, QueryInstance]()
    query_kinds = dict[DriverType, QttMapping]()
    for driver_type in DriverType:
        queries_for_type = generate_queries(driver_type, SCHEMA_ID)
        queires.update(queries_for_type)
        query_kinds[driver_type] = extract_query_kinds(driver_type, queries_for_type)

    query_engine = QueryEngine(estimator, queires, query_weights)
    output_collector = OutputCollector(query_kinds)

    dbs = [DriverType.POSTGRES.value, DriverType.MONGO.value, DriverType.NEO4J.value]

    NUM_QUERIES = 11
    kinds = list(range(NUM_QUERIES))
    initial_mapping = [DriverType.POSTGRES.value] * len(kinds)
    mcts = MCTS(query_engine, kinds, output_collector, dbs)

    # Run MCTS to find the best execution plan
    mcts.run(initial_mapping, MAX_ITERATIONS)

def get_query_id(driver_type: DriverType, query_index: int) -> QueryInstanceId:
    # FIXME Not ideal. We should allow passing arbitrary queries. But that would require to group all generated queries by template_name, and then store them in the state by template_name.
    template_name = f"edbt-{query_index}"

    database_id = create_database_id(driver_type, *parse_schema_id(SCHEMA_ID))
    return create_query_instance_id(database_id, template_name, 0)

def load_driver_type(ctx: Context, driver_type: DriverType):
    dataset_id = create_dataset_id(driver_type, DATASET_NAME)
    feature_extractor = load_feature_extractor(ctx.pp.feature_extractor(dataset_id))

    checkpoint_id = create_checkpoint_id(create_model_id(driver_type, MODEL_NAME), CHECKPOINT_NAME)
    model = ctx.mp.load_model(ctx.pp.model(checkpoint_id))

    return (driver_type, feature_extractor, model)

def load_database(ctx: Context, driver_type: DriverType, schema_id: SchemaId):
    schema_name, scale = parse_schema_id(schema_id)
    database_id = create_database_id(driver_type, schema_name, scale)

    driver = ctx.dp.get(driver_type, schema_name, scale)
    plan_extractor = get_plan_extractor(driver)

    return (database_id, plan_extractor)

def generate_queries(driver_type: DriverType, schema_id: SchemaId) -> dict[QueryInstanceId, QueryInstance]:
    schema, scale = parse_schema_id(schema_id)
    registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
    instances = registry.generate_queries(scale, 0)
    return {instance.id: instance for instance in instances}

class QueryEngine:
    def __init__(self,
        latency_estimator: LatencyEstimator,
        queries: dict[QueryInstanceId, QueryInstance],
        query_weights: list[float]
    ):
        self.latency_estimator = latency_estimator
        self.queries = queries
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
            driver_type = DriverType(state[query_index])
            query_id = get_query_id(driver_type, query_index)

            query = self.queries[query_id]
            latency = self.latency_estimator.estimate(query)
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

def extract_query_kinds(driver_type: DriverType, queries: dict[QueryInstanceId, QueryInstance]) -> QttMapping:
    extractor = get_kind_extractor(driver_type)
    return {query_id: extractor.extract_query_kinds(query) for query_id, query in queries.items()}

class OutputCollector:
    def __init__(self, query_kinds: dict[DriverType, QttMapping]):
        self.query_kinds = query_kinds
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
        schema = dict[DriverType, set[str]]()
        for query_index, driver_type_str in enumerate(state):
            driver_type = DriverType(driver_type_str)
            query_id = get_query_id(driver_type, query_index)

            if driver_type not in schema:
                schema[driver_type] = set()

            kinds = self.query_kinds[driver_type].get(query_id, [])

            schema[driver_type].update(kinds)

        return {driver_type.value: list(kinds) for driver_type, kinds in schema.items()}

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
