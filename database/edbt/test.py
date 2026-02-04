import argparse
from common.utils import auto_close

NUM_RUNS = 1

def main():
    parser = argparse.ArgumentParser(description='Test Postgres EDBT')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to trained model')
    parser.add_argument('--database', '-d', type=str, required=True, help='Either "postgres" or "neo4j"')

    args = parser.parse_args()

    if args.database == "postgres":
        evaluate_postgres(args.checkpoint)
    elif args.database == "neo4j":
        evaluate_neo4j(args.checkpoint)
    else:
        print(f'Unsupported database type: {args.database}')

def evaluate_postgres(checkpoint: str):
    from datasets.edbt.postgres_database import EdbtPostgresDatabase
    from latency_estimation.postgres.context import PostgresContext
    from latency_estimation.postgres.latency_estimator import LatencyEstimator

    ctx = PostgresContext.create(database=EdbtPostgresDatabase())
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database.get_test_queries():
            try:
                print(f'Executing query {query.name}...')
                estimated, _ = estimator.estimate(query.content)
                actual, _, _, num_results = ctx.extractor.measure_query(query.content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except Exception as e:
                print(f'Error: {e}\n')

def evaluate_neo4j(checkpoint: str):
    from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
    from latency_estimation.neo4j.context import Neo4jContext
    from latency_estimation.neo4j.latency_estimator import LatencyEstimator

    ctx = Neo4jContext.create(database=EdbtNeo4jDatabase())
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database.get_test_queries():
            try:
                print(f'Executing query {query.name}...')
                estimated, _ = estimator.estimate(query.content)
                actual, _, num_results = ctx.extractor.measure_query(query.content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except Exception as e:
                print(f'Error: {e}\n')

def print_query_results(num_results: int, estimated: float, actual: float):
    print(f'Returned {num_results} rows. Estimated: {estimated * 1000:.2f} ms, Actual: {actual * 1000:.2f} ms\n')

if __name__ == '__main__':
    main()
