import json
import time
import random
from config import DatabaseConfig

class PlanExtractor:
    """Extracts query plans and execution statistics from PostgreSQL."""

    def __init__(self, config: DatabaseConfig):
        self.config = config

    def execute_with_plan(self, query: str, clear_cache: bool = True) -> tuple[dict, float]:
        """
        Execute a query and return its plan and actual execution time.

        Args:
            query: SQL query to execute
            clear_cache: Whether to clear PostgreSQL cache before execution

        Returns:
            Tuple of (plan_dict, execution_time_ms)
        """
        conn = self.config.get_connection()

        try:
            # Set autocommit to avoid transaction block issues
            conn.autocommit = True

            with conn.cursor() as cursor:
                # Clear cache if requested (simulates cold cache)
                if clear_cache:
                    try:
                        cursor.execute("DISCARD ALL;")
                    except Exception as e:
                        # If DISCARD ALL fails, try alternative cache clearing
                        print(f"Warning: Could not clear cache: {e}")
                        pass

                # Get the plan with execution statistics
                explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}"

                cursor.execute(explain_query)
                result = cursor.fetchone()

                # Parse JSON plan
                plan_json = result[0][0]  # EXPLAIN returns list of plans

                # Extract actual execution time from plan
                execution_time = plan_json['Execution Time']  # in ms

                return plan_json, execution_time

        finally:
            conn.close()

    def generate_tpch_queries(self, num_queries: int = 100, scale_factor: int = 1) -> list[str]:
        """
        Generate TPC-H style queries with parameter variations.

        Args:
            num_queries: Number of queries to generate
            scale_factor: TPC-H scale factor (affects data size)

        Returns:
            List of SQL query strings
        """
        queries = []

        # TPC-H Query 1 variations
        for _ in range(num_queries // 6):
            delta = random.randint(60, 120)
            queries.append(f"""
                SELECT
                    l_returnflag,
                    l_linestatus,
                    SUM(l_quantity) as sum_qty,
                    SUM(l_extendedprice) as sum_base_price,
                    SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
                    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
                    AVG(l_quantity) as avg_qty,
                    AVG(l_extendedprice) as avg_price,
                    AVG(l_discount) as avg_disc,
                    COUNT(*) as count_order
                FROM lineitem
                WHERE l_shipdate <= date '1998-12-01' - interval '{delta} days'
                GROUP BY l_returnflag, l_linestatus
                ORDER BY l_returnflag, l_linestatus
            """)

        # TPC-H Query 3 variations
        for _ in range(num_queries // 6):
            segment = random.choice(['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'])
            date = f"1995-03-{random.randint(1, 31):02d}"
            queries.append(f"""
                SELECT
                    l_orderkey,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue,
                    o_orderdate,
                    o_shippriority
                FROM customer, orders, lineitem
                WHERE c_mktsegment = '{segment}'
                    AND c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND o_orderdate < date '{date}'
                    AND l_shipdate > date '{date}'
                GROUP BY l_orderkey, o_orderdate, o_shippriority
                ORDER BY revenue DESC, o_orderdate
                LIMIT 10
            """)

        # TPC-H Query 5 variations
        for _ in range(num_queries // 6):
            year = random.randint(1993, 1997)
            queries.append(f"""
                SELECT
                    SUM(l_extendedprice * (1 - l_discount)) as revenue
                FROM customer, orders, lineitem, supplier
                WHERE c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND l_suppkey = s_suppkey
                    AND c_nationkey = s_nationkey
                    AND o_orderdate >= date '{year}-01-01'
                    AND o_orderdate < date '{year}-01-01' + interval '1 year'
                GROUP BY c_nationkey
                ORDER BY revenue DESC
            """)

        # TPC-H Query 6 variations
        for _ in range(num_queries // 6):
            year = random.randint(1993, 1997)
            discount = random.uniform(0.02, 0.09)
            quantity = random.randint(20, 30)
            queries.append(f"""
                SELECT
                    SUM(l_extendedprice * l_discount) as revenue
                FROM lineitem
                WHERE l_shipdate >= date '{year}-01-01'
                    AND l_shipdate < date '{year}-01-01' + interval '1 year'
                    AND l_discount BETWEEN {discount} - 0.01 AND {discount} + 0.01
                    AND l_quantity < {quantity}
            """)

        # TPC-H Query 10 variations
        for _ in range(num_queries // 6):
            year = random.randint(1993, 1997)
            month = random.randint(1, 12)
            queries.append(f"""
                SELECT
                    c_custkey,
                    c_name,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue,
                    c_acctbal,
                    c_address,
                    c_phone,
                    c_comment
                FROM customer, orders, lineitem
                WHERE c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND o_orderdate >= date '{year}-{month:02d}-01'
                    AND o_orderdate < date '{year}-{month:02d}-01' + interval '3 months'
                    AND l_returnflag = 'R'
                GROUP BY c_custkey, c_name, c_acctbal, c_phone, c_address, c_comment
                ORDER BY revenue DESC
                LIMIT 20
            """)

        # TPC-H Query 12 variations
        for _ in range(num_queries // 6):
            year = random.randint(1993, 1997)
            mode1 = random.choice(['MAIL', 'SHIP', 'AIR', 'TRUCK'])
            mode2 = random.choice(['RAIL', 'FOB', 'REG AIR'])
            queries.append(f"""
                SELECT
                    l_shipmode,
                    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
                        THEN 1 ELSE 0 END) as high_line_count,
                    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
                        THEN 1 ELSE 0 END) as low_line_count
                FROM orders, lineitem
                WHERE o_orderkey = l_orderkey
                    AND l_shipmode IN ('{mode1}', '{mode2}')
                    AND l_commitdate < l_receiptdate
                    AND l_shipdate < l_commitdate
                    AND l_receiptdate >= date '{year}-01-01'
                    AND l_receiptdate < date '{year}-01-01' + interval '1 year'
                GROUP BY l_shipmode
                ORDER BY l_shipmode
            """)

        return queries[:num_queries]

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
        queries = self.generate_tpch_queries(num_queries)
        dataset = []

        print(f"Collecting {len(queries)} query plans...")

        if not clear_cache:
            print("Note: Cache clearing is disabled for faster collection.")
            print("      Set clear_cache=True for cold-cache measurements.\n")

        for i, query in enumerate(queries):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(queries)} ({100*i//len(queries)}%)")

            try:
                plan, exec_time = self.execute_with_plan(query, clear_cache=clear_cache)
                dataset.append({
                    'query': query,
                    'plan': plan['Plan'],
                    'execution_time': exec_time
                })
            except Exception as e:
                print(f"\nError executing query {i}: {e}")
                print(f"Query preview: {query[:100]}...\n")
                continue

        print(f"\nCollected {len(dataset)} query plans successfully")
        return dataset
