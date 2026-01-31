from datasets.database import Database
from common.drivers import PostgresDriver

class PlanExtractor:
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, postgres: PostgresDriver, database: Database):
        self.postgres = postgres
        self.database = database

    def execute_with_plan(self, query: str, clear_cache: bool = True) -> tuple[dict, float]:
        """
        Execute a query and return its plan and actual execution time.

        Args:
            query: SQL query to execute
            clear_cache: Whether to clear PostgreSQL cache before execution

        Returns:
            Tuple of (plan_dict, execution_time_ms)
        """
        connection = self.postgres.get_connection()
        try:
            # Set autocommit to avoid transaction block issues
            connection.autocommit = True

            with connection.cursor() as cursor:
                # Clear cache if requested (simulates cold cache)
                if clear_cache:
                    try:
                        cursor.execute('DISCARD ALL;')
                    except Exception as e:
                        # If DISCARD ALL fails, try alternative cache clearing
                        print(f'Warning: Could not clear cache: {e}')
                        pass

                # Get the plan with execution statistics
                explain_query = f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}'

                cursor.execute(explain_query)
                result = cursor.fetchone()

                if result is None:
                    raise RuntimeError('No plan returned from EXPLAIN.')

                # Parse JSON plan
                plan_json = result[0][0]  # EXPLAIN returns list of plans

                # Extract actual execution time from plan
                execution_time = plan_json['Execution Time']  # in ms

                return plan_json, execution_time

        finally:
            self.postgres.put_connection(connection)

    def collect_training_data(self, num_queries: int = 1000, clear_cache: bool = True) -> list[dict]:
        """
        Collect a dataset of query plans and execution times.

        Args:
            num_queries: Number of queries to collect
            clear_cache: Whether to clear cache before each query (slower but more realistic)

        Returns:
            List of dictionaries containing:
            - query: SQL query string
            - plan: Parsed query plan tree
            - execution_time: Actual execution time in ms
        """
        queries = self.database.get_train_queries(num_queries)
        dataset = []

        print(f'Collecting {len(queries)} query plans...')

        if not clear_cache:
            print('Note: Cache clearing is disabled for faster collection.')
            print('      Set clear_cache=True for cold-cache measurements.\n')

        for i, query in enumerate(queries):
            if i % 50 == 0:
                print(f'Progress: {i}/{len(queries)} ({100*i//len(queries)}%)')

            try:
                plan, exec_time = self.execute_with_plan(query, clear_cache=clear_cache)
                dataset.append({
                    'query': query,
                    'plan': plan['Plan'],
                    'execution_time': exec_time
                })
            except Exception as e:
                print(f'\nError executing query {i}: {e}')
                print(f'Query preview: {query[:100]}...\n')
                continue

        print(f'\nCollected {len(dataset)} query plans successfully')
        return dataset
