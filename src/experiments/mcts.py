from dataclasses import dataclass
import os
import random
import time
from core.config import Config
from core.drivers import DriverType
from core.files import open_input, open_output
from core.loaders.base_loader import load_populate_times
from core.query import create_database_id_2
from core.utils import ProgressTracker, exit_with_error, print_warning
from providers.path_provider import PathProvider
from search.mcts import MCTS, StateMapping
from search.run import AdaptationSchema
# FIXME Use measure_queries
from experiments.measure import load_database_measurement
from experiments.plot_measurement import SCALES

# TODO unify with common.config
NUM_QUERIES = 11
MAX_ITERATIONS = 500000
ITERATION_CAPTURE_INTERVAL = 1000
IS_VERBOSE = False
SCHEMA = 'TODO' # FIXME

def main():
    config = Config.load()

    # parser = argparse.ArgumentParser(description='Run experiments.')
    # parser.add_argument('query-weights', nargs='*', type=float, help=f'Query weights for the test queries.')
    # parser.add_argument('--scale', type=float, required=True, help='Scale factor of the generated schema data.')
    # args = parser.parse_args()
    # scale: float = args.scale

    # for i, scale in enumerate(SCALES):
    #     # Run the larger scales more times since they should be more interesting.
    #     for _ in range(i + 1):
    #         run_id = int(time.time())
    #         do_run(config, run_id, scale)
    #         print()

    for _ in range(10):
        run_id = int(time.time())
        do_run(config, run_id, 1)
        print()

def do_run(config: Config, run_id: int, scale: float):
    rng = random.Random(run_id)

    # query_weights: list[float] = args.query_weights
    query_weights: list[float] = [
        rng.randint(20, 80), # test:0, M
        rng.randint(10, 40), # test:1, M
        rng.randint( 2,  8), # test:2, M
        rng.randint( 1,  4), # test:3
        rng.randint( 5, 20), # test:4
        rng.randint( 1,  4), # test:5,
        rng.randint( 1,  4), # test:6, M
        rng.randint( 2,  8), # test:7
        rng.randint( 1,  4), # test:8
        rng.randint(10, 40), # test:9, M
        rng.randint( 5, 20), # test:10
    ]
    if len(query_weights) != NUM_QUERIES:
        exit_with_error(f'Error: Expected {NUM_QUERIES} query weights, but got {len(query_weights)}.')

    migration_coefficient = float(f'{rng.uniform(0.03, 0.1):.3f}')

    dbs = [DriverType.POSTGRES.value, DriverType.MONGO.value, DriverType.NEO4J.value]
    kinds = list(range(NUM_QUERIES))
    initial_mapping = [DriverType.POSTGRES.value] * len(kinds)

    print('Starting MCTS with:')
    print(f'  id: {run_id}')
    print(f'  schema: {SCHEMA.value}')
    print(f'  scale: {scale}')
    print(f'  query weights: {query_weights}')
    print(f'  initial mapping: {initial_mapping}')
    print(f'  migration latency coefficient: {migration_coefficient}')
    print()

    query_engine = create_query_engine(config, scale, query_weights, migration_coefficient)
    progress_tracker = ProgressTracker.limited(MAX_ITERATIONS)
    output_collector = OutputCollector(query_engine, progress_tracker)
    mcts = MCTS(query_engine, kinds, output_collector, dbs)

    progress_tracker.start('Running MCTS... ')
    mcts.run(initial_mapping, MAX_ITERATIONS)
    progress_tracker.finish()

    save_stats(config, run_id, output_collector, scale)

QueryCosts = list[float]
KindCosts = dict[str, float]

def create_query_engine(config: Config, scale: float, query_weights: list[float], migration_coefficient: float) -> 'QueryEngine':
    pp = PathProvider(config)

    times = dict[DriverType, QueryCosts]()
    migration_times = dict[DriverType, KindCosts]()

    for type in DriverType:
        database_id = create_database_id_2(type, SCHEMA, scale)
        results = load_database_measurement(config, info)

        times_by_query = QueryCosts()
        times[type] = times_by_query

        for result in results:
            times_by_query.append(result.mean)

        migration_times[type] = load_populate_times(pp.populate_times(database_id))

    return QueryEngine(times, migration_times, migration_coefficient, query_weights)

class QueryEngine:
    def __init__(self, times: dict[DriverType, QueryCosts], migration_times: dict[DriverType, KindCosts], migration_coefficient: float, query_weights: list[float]):
        self.times = times
        self.migration_times = migration_times
        self.migration_coefficient = migration_coefficient
        self.query_weights = query_weights
        self.query_kinds = extract_query_kinds()

    def estimate_latency(self, state: StateMapping) -> float:
        return self.estimate_query_latency(state) + self.estimate_migration_latency(state)

    def estimate_query_latency(self, state: StateMapping) -> float:
        total = 0.0
        for query_index, weight in enumerate(self.query_weights):
            type = DriverType(state[query_index])
            latency = self.times[type][query_index]
            total += float(latency) * weight

        return total

    def estimate_migration_latency(self, state: StateMapping) -> float:
        kinds_by_db: dict[DriverType, set[str]] = {
            DriverType.POSTGRES: set(),
            DriverType.MONGO: set(),
            DriverType.NEO4J: set(),
        }

        for query_index in range(len(self.query_weights)):
            type = DriverType(state[query_index])
            kinds = self.query_kinds[type][query_index]
            kinds_by_db[type].update(kinds)

        total = 0.0
        for type, kinds in kinds_by_db.items():
            for kind in kinds:
                total += self.migration_times[type][kind]

        return total * self.migration_coefficient

@dataclass
class AdaptationSolution:
    id: int
    latency: float
    query_latency: float
    migration_latency: float
    found_in: float
    objexes: AdaptationSchema

IterationStats = tuple[float, int, int]
"""[time [s], iteration number, processed states]"""

class OutputCollector:
    def __init__(self, engine: QueryEngine, progress_tracker: ProgressTracker):
        self.engine = engine
        self.solutions = list[AdaptationSolution]()
        self.iterations = list[IterationStats]()
        self.progress_tracker = progress_tracker

    def on_initial_solution(self, state: StateMapping, latency: float):
        self.start = time.perf_counter()
        self.iteration_counter = 0

        solution = self.__convert_state_to_solution(state, latency, 0)
        self.solutions.append(solution)
        self.try_print_solution(solution)

    def on_best_solution(self, state: StateMapping, latency: float, processed_states: int):
        solution = self.__convert_state_to_solution(state, latency, processed_states)
        self.solutions.append(solution)
        self.try_print_solution(solution)

    def __convert_state_to_solution(self, state: StateMapping, latency: float, processed_states: int) -> AdaptationSolution:
        query_latency = self.engine.estimate_query_latency(state)
        migration_latency = self.engine.estimate_migration_latency(state)

        return AdaptationSolution(
            id=processed_states,
            latency=latency,
            query_latency=query_latency,
            migration_latency=migration_latency,
            found_in=time.perf_counter() - self.start,
            objexes=self.__convert_state_to_schema(state),
        )

    def __convert_state_to_schema(self, state: StateMapping) -> AdaptationSchema:
        schema = dict[str, set[str]]()
        for query_index, db in enumerate(state):
            if db not in schema:
                schema[db] = set()

            type = DriverType(db)
            kinds = self.engine.query_kinds[type][query_index]
            schema[db].update(kinds)

        return {db: list(kinds) for db, kinds in schema.items()}

    def try_print_solution(self, solution: AdaptationSolution):
        if not IS_VERBOSE:
            return

        print(f'\n[{solution.found_in:.3f} s] #{solution.id}')
        print(f'Latency {solution.latency:.2f} ms (query: {solution.query_latency:.2f} ms, migration: {solution.migration_latency:.2f} ms)')
        for db, kinds in solution.objexes.items():
            print(f'  {db}: {", ".join(kinds)}')

    def on_iteration(self, iteration: int, processed_states: int):
        self.iteration_counter += 1
        if self.iteration_counter == ITERATION_CAPTURE_INTERVAL:
            self.iteration_counter = 0
            elapsed_s = time.perf_counter() - self.start
            self.iterations.append((elapsed_s, iteration, processed_states))
            self.progress_tracker.track(ITERATION_CAPTURE_INTERVAL)

            # print(f'\n[{elapsed_s:.3f} s] iterations: {iteration}, states: {processed_states}')

def save_stats(config: Config, run_id: int, output_collector: OutputCollector, scale: float):
    iterations_path = os.path.join(config.experiments_directory, f'mcts_{SCHEMA.value}_{scale:g}_{run_id}_iterations.csv')
    with open_output(iterations_path, skip_dir_check=True) as file:
        file.write('time,iteration,processed_states\n')
        for stats in output_collector.iterations:
            file.write(f'{stats[0]:.3f},{stats[1]},{stats[2]}\n')

    solutions_path = os.path.join(config.experiments_directory, f'mcts_{SCHEMA.value}_{scale:g}_{run_id}_solutions.csv')
    with open_output(solutions_path, skip_dir_check=True) as file:
        header = 'found_in,id,latency,query_latency,migration_latency'
        for db in DriverType:
            header += f',{db.value}'
        file.write(header + '\n')

        for solution in output_collector.solutions:
            row = f'{solution.found_in:.3f},{solution.id},{solution.latency:.3f},{solution.query_latency:.3f},{solution.migration_latency:.3f}'
            for db in DriverType:
                kinds = solution.objexes.get(db.value, [])
                row += f',{len(kinds)}'
            file.write(row + '\n')

@dataclass
class LoadedAdaptationSolution:
    id: int
    latency: float
    query_latency: float
    migration_latency: float
    found_in: float
    postgres: int
    mongo: int
    neo4j: int

def load_stats(config: Config, run_id: int, scale: float) -> tuple[list[IterationStats], list[LoadedAdaptationSolution]]:
    iterations = list[IterationStats]()

    iterations_path = os.path.join(config.experiments_directory, f'mcts_{SCHEMA.value}_{scale:g}_{run_id}_iterations.csv')
    with open_input(iterations_path) as file:
        next(file) # skip header
        for line in file:
            parts = line.strip().split(',')
            iterations.append((float(parts[0]), int(parts[1]), int(parts[2])))

    solutions = list[LoadedAdaptationSolution]()

    solutions_path = os.path.join(config.experiments_directory, f'mcts_{SCHEMA.value}_{scale:g}_{run_id}_solutions.csv')
    with open_input(solutions_path) as file:
        next(file) # skip header
        for line in file:
            parts = line.strip().split(',')
            solution = LoadedAdaptationSolution(
                found_in=float(parts[0]),
                id=int(parts[1]),
                latency=float(parts[2]),
                query_latency=float(parts[3]),
                migration_latency=float(parts[4]),
                postgres=int(parts[5]),
                mongo=int(parts[6]),
                neo4j=int(parts[7]),
            )
            solutions.append(solution)

    return iterations, solutions

def find_stats(config: Config) -> list[tuple[float, int]]:
    """Returns tuples of (scale, run_id)."""
    output = list[tuple[float, int]]()

    files = os.listdir(config.experiments_directory)
    for file in files:
        if not file.startswith(f'mcts_{SCHEMA.value}_') or not file.endswith('_iterations.csv'):
            continue

        parts = file[len(f'mcts_{SCHEMA.value}_'):-len('_iterations.csv')].split('_')
        if len(parts) != 2:
            continue

        try:
            scale = float(parts[0])
            run_id = int(parts[1])
            output.append((scale, run_id))
        except ValueError as e:
            print_warning(f'Could not parse scale and run_id from file name: {file}', e)

    return output

QueryRequiredKinds = list[set[str]]

# FIXME Infer this automatically - not sure the current implementation works tho.
def extract_query_kinds() -> dict[DriverType, QueryRequiredKinds]:
    output = {
        DriverType.POSTGRES: [
            {'order', 'customer'},
            {'order', 'customer', 'order_item', 'product'},
            {'order', 'customer', 'order_item'},
            {'order', 'order_item', 'product'},
            {'order', 'order_item', 'product', 'has_category'},
            {'order', 'customer'},
            {'order', 'customer', 'order_item', 'product'},
            {'follows'},
            {'product', 'has_category', 'has_interest'},
            {'product', 'seller', 'review'},
            {'order', 'order_item'},
        ],
        DriverType.NEO4J: [
            {'Person', 'Customer', 'Order', 'SNAPSHOT_OF', 'PLACED'},
            {'Person', 'Customer', 'Order', 'Product', 'SNAPSHOT_OF', 'PLACED', 'HAS_ITEM'},
            {'Person', 'Customer', 'Order', 'Product', 'SNAPSHOT_OF', 'PLACED', 'HAS_ITEM'},
            {'Seller', 'Product', 'Order', 'OFFERS', 'HAS_ITEM'},
            {'Category', 'Product', 'Order', 'HAS_CATEGORY', 'HAS_ITEM'},
            {'Person', 'Customer', 'Order', 'SNAPSHOT_OF', 'PLACED'},
            {'Person', 'Customer', 'Order', 'Product', 'Seller', 'SNAPSHOT_OF', 'PLACED', 'HAS_ITEM', 'OFFERS'},
            {'Person', 'FOLLOWS'},
            {'Person', 'Category', 'Product', 'HAS_INTEREST', 'HAS_CATEGORY'},
            {'Product', 'Seller', 'Customer', 'OFFERS', 'REVIEWED'},
            {'Product', 'Order', 'Product', 'HAS_IT'},
        ],
        DriverType.MONGO: [
            {'order'},
            {'order'},
            {'order'},
            {'order'},
            {'order', 'product'},
            {'order'},
            {'order'},
            {'person'},
            {'person', 'product'},
            {'product'},
            {'order'},
        ],
    }

    for _, kinds in output.items():
        if len(kinds) != NUM_QUERIES:
            exit_with_error(f'Error: Expected {NUM_QUERIES} query kinds, but got {len(kinds)} for database.')

    return output

if __name__ == "__main__":
    main()
