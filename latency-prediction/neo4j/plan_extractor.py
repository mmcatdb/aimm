"""
Extract query execution plans and execution times from Neo4j.
This module handles:
- Generating TPC-H query variants with random parameters
- Executing queries with EXPLAIN (for plan) and timing actual execution
- Parsing the plan structure
- Recording ground truth latencies
"""
import yaml
import time
import datetime
import random
from typing import Any
from neo4j import GraphDatabase
import numpy as np

class PlanExtractor:
    """
    Extracts query plans and execution times from Neo4j.
    Generates multiple query variants by substituting parameters.
    """

    NUM_QUERY_TYPES = 32

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize connection to Neo4j database.

        Args:
            config_path: Path to YAML config file with Neo4j credentials
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        neo4j_config = config['neo4j']
        self.driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()

    def generate_workload_queries(self, num_queries: int = 350) -> list[str]:
        """
        Generate TPC-H query variants with random parameters.
        Similar to PostgreSQL implementation but for Cypher queries.

        Args:
            num_queries: Total number of queries to generate

        Returns:
            List of query strings
        """
        queries = []
        # This will now be smaller, distributing the total num_queries over all 32 types
        queries_per_type = num_queries // self.NUM_QUERY_TYPES

        # --- Define constant lists for parameters (from PostgreSQL version) ---
        SHIPMODES_ALL = ['MAIL', 'SHIP', 'AIR', 'TRUCK', 'RAIL', 'FOB', 'REG AIR']
        REGION_CHOICES = ['ASIA', 'AMERICA', 'EUROPE', 'MIDDLE EAST', 'AFRICA']
        SEGMENT_CHOICES = ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
        NATIONS = ["ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA",
                       "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ",
                       "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU",
                       "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"]
        P_NAME_WORDS = [
            "almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue",
            "blush", "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral",
            "cornflower", "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick",
            "floral", "forest", "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew",
            "hot", "indian", "ivory", "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen",
            "magenta", "maroon", "medium", "metallic", "midnight", "mint", "misty", "moccasin", "navajo",
            "navy", "olive", "orange", "orchid", "pale", "papaya", "peach", "peru", "pink", "plum", "powder",
            "puff", "purple", "red", "rose", "rosy", "royal", "saddle", "salmon", "sandy", "seashell", "sienna",
            "sky", "slate", "smoke", "snow", "spring", "steel", "tan", "thistle", "tomato", "turquoise", "violet",
            "wheat", "white", "yellow"
        ]
        CONTAINER_CHOICES_SM = ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']
        CONTAINER_CHOICES_LG = ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
        ORDER_STATUS_CHOICES = ['F', 'O', 'P']
        ORDER_PRIORITY_CHOICES = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']


        def get_random_name(table_name, max_id, min_id=1, id_length=9):
            num = random.randint(min_id, max_id)
            num_padding_zeroes = id_length - len(str(num))
            return f'{table_name}#{'0'*num_padding_zeroes}{num}'


        # Q1: Pricing Summary Report
        for _ in range(queries_per_type):
            delta = random.randint(60, 120)
            base_date = datetime.date(1998, 12, 1)
            target_date = base_date - datetime.timedelta(days=delta)
            date_str = target_date.strftime('%Y-%m-%d')

            queries.append(f"""
MATCH (li:LineItem)
WHERE li.l_shipdate <= date('{date_str}')
WITH li.l_returnflag AS returnflag, li.l_linestatus AS linestatus, li
RETURN
  returnflag,
  linestatus,
  sum(li.l_quantity) AS sum_qty,
  sum(li.l_extendedprice) AS sum_base_price,
  sum(li.l_extendedprice * (1 - li.l_discount)) AS sum_disc_price,
  sum(li.l_extendedprice * (1 - li.l_discount) * (1 + li.l_tax)) AS sum_charge,
  avg(li.l_quantity) AS avg_qty,
  avg(li.l_extendedprice) AS avg_price,
  avg(li.l_discount) AS avg_disc,
  count(li) AS count_order
ORDER BY returnflag, linestatus
""")

        # Q5: Local Supplier Volume
        for _ in range(queries_per_type):
            region = random.choice(REGION_CHOICES)
            year = random.randint(1993, 1997)
            start_date = f"{year}-01-01"
            end_date = f"{year+1}-01-01"

            queries.append(f"""
MATCH (r:Region {{r_name: '{region}'}})<-[:IS_IN_REGION]-(n:Nation)
MATCH (n)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
MATCH (n)<-[:IS_IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li)
WHERE o.o_orderdate >= date('{start_date}')
  AND o.o_orderdate < date('{end_date}')
WITH n.n_name AS nation_name,
     sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
RETURN nation_name, revenue
ORDER BY revenue DESC
""")

        # Q6: Forecasting Revenue Change
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)

            discount_center = random.uniform(0.02, 0.09)
            discount_low = discount_center - 0.01
            discount_high = discount_center + 0.01

            quantity = random.randint(24, 25)

            queries.append(f"""
MATCH (li:LineItem)
WHERE li.l_shipdate >= date('{year}-01-01')
  AND li.l_shipdate < date('{year+1}-01-01')
  AND li.l_discount >= {discount_low:.2f}
  AND li.l_discount <= {discount_high:.2f}
  AND li.l_quantity < {quantity}
RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
""")

        # Q10: Returned Item Reporting (Fix: Date generation)
        for _ in range(queries_per_type):
            random_month_offset = random.randint(0, 23)  # 24 possible months (Feb 1993 to Jan 1995)
            year = 1993 + ((random_month_offset + 1) // 12)
            month = ((random_month_offset + 1) % 12) + 1
            start_date = f"{year}-{month:02d}-01"

            # Add 3 months
            end_month = month + 3
            end_year = year
            if end_month > 12:
                end_month -= 12
                end_year += 1
            end_date = f"{end_year}-{end_month:02d}-01"

            queries.append(f"""
MATCH (n:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
WHERE o.o_orderdate >= date('{start_date}')
  AND o.o_orderdate < date('{end_date}')
  AND li.l_returnflag = 'R'
WITH c, n, sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
RETURN
  c.c_custkey AS custkey,
  c.c_name AS name,
  revenue,
  c.c_acctbal AS acctbal,
  n.n_name AS nation,
  c.c_address AS address,
  c.c_phone AS phone,
  c.c_comment AS comment
ORDER BY revenue DESC
LIMIT 20
""")

        # Q12: Shipping Modes and Order Priority (Fix: Shipmode generation)
        for _ in range(queries_per_type):
            shipmodes = random.sample(SHIPMODES_ALL, 2)
            year = random.randint(1993, 1997)

            queries.append(f"""
MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
WHERE li.l_shipmode IN {shipmodes}
  AND li.l_commitdate < li.l_receiptdate
  AND li.l_shipdate < li.l_commitdate
  AND li.l_receiptdate >= date('{year}-01-01')
  AND li.l_receiptdate < date('{year+1}-01-01')
WITH li.l_shipmode AS shipmode, o.o_orderpriority AS priority
RETURN
  shipmode,
  sum(CASE
    WHEN priority = '1-URGENT' OR priority = '2-HIGH'
    THEN 1
    ELSE 0
  END) AS high_priority_count,
  sum(CASE
    WHEN priority <> '1-URGENT' AND priority <> '2-HIGH'
    THEN 1
    ELSE 0
  END) AS low_priority_count
ORDER BY shipmode
""")

        # Q14: Promotion Effect (Fix: Month range and end date logic)
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            month = random.randint(1, 12)  # Fix: Was 1-11
            start_date = f"{year}-{month:02d}-01"

            end_month = month + 1
            end_year = year
            if end_month > 12:
                end_month = 1
                end_year += 1
            end_date = f"{end_year}-{end_month:02d}-01"

            queries.append(f"""
MATCH (li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
WHERE li.l_shipdate >= date('{start_date}')
  AND li.l_shipdate < date('{end_date}')
WITH sum(
  CASE
    WHEN p.p_type STARTS WITH 'PROMO'
    THEN li.l_extendedprice * (1 - li.l_discount)
    ELSE 0
  END
) AS promo_revenue,
sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue
RETURN 100.00 * promo_revenue / total_revenue AS promo_revenue_percentage
""")

        # Q19: Discounted Revenue (Fix: Brand generation)
        for _ in range(queries_per_type):
            brand1 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
            brand2 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
            brand3 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"

            qty1 = random.randint(1, 10)
            qty2 = random.randint(10, 20)
            qty3 = random.randint(20, 30)

            queries.append(f"""
MATCH (li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
WHERE li.l_shipinstruct = 'DELIVER IN PERSON'
  AND li.l_shipmode IN ['AIR', 'AIR REG']
  AND (
    (
      p.p_brand = '{brand1}'
      AND p.p_container IN ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']
      AND li.l_quantity >= {qty1} AND li.l_quantity <= {qty1+10}
      AND p.p_size >= 1 AND p.p_size <= 5
    ) OR (
      p.p_brand = '{brand2}'
      AND p.p_container IN ['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK']
      AND li.l_quantity >= {qty2} AND li.l_quantity <= {qty2+10}
      AND p.p_size >= 1 AND p.p_size <= 10
    ) OR (
      p.p_brand = '{brand3}'
      AND p.p_container IN ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
      AND li.l_quantity >= {qty3} AND li.l_quantity <= {qty3+10}
      AND p.p_size >= 1 AND p.p_size <= 15
    )
  )
RETURN sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
""")

        # Custom Q1: Simple Scan (All Nodes of a Label)
        for _ in range(queries_per_type):
            queries.append("MATCH (n:Nation) RETURN n.n_name, n.n_comment LIMIT 100")

        # Custom Q2: Scan with Exact Property Match
        for _ in range(queries_per_type):
            name = get_random_name('Customer', 30000)
            queries.append(f"MATCH (c:Customer {{c_name: '{name}'}}) RETURN c.c_address, c.c_phone")

        # Custom Q3: Scan with Numeric Filter
        for _ in range(queries_per_type):
            balance = random.randint(5000, 9500)
            queries.append(f"MATCH (s:Supplier) WHERE s.s_acctbal > {balance} RETURN s.s_name, s.s_acctbal")

        # Custom Q4: Scan with Sort
        for _ in range(queries_per_type):
            queries.append("MATCH (p:Part) RETURN p.p_name, p.p_retailprice ORDER BY p.p_retailprice DESC LIMIT 50")

        # Custom Q5: Scan with Sort + Limit
        for _ in range(queries_per_type):
            limit = random.randint(10, 50)
            queries.append(f"MATCH (o:Order) RETURN o.o_orderkey, o.o_totalprice ORDER BY o.o_totalprice DESC LIMIT {limit}")

        # Custom Q6: Scan with IN list
        for _ in range(queries_per_type):
            containers = random.sample(CONTAINER_CHOICES_SM + CONTAINER_CHOICES_LG, 3)
            queries.append(f"MATCH (p:Part) WHERE p.p_container IN {containers} RETURN p.p_name, p.p_container")

        # Custom Q7: Scan with STARTS WITH
        TYPE_CHOICES = [
            'STANDARD ANODIZED', 'STANDARD BURNISHED', 'STANDARD PLATED', 'STANDARD POLISHED', 'STANDARD BRUSHED',
            'SMALL ANODIZED', 'SMALL BURNISHED', 'SMALL PLATED', 'SMALL POLISHED', 'SMALL BRUSHED',
            'MEDIUM ANODIZED', 'MEDIUM BURNISHED', 'MEDIUM PLATED', 'MEDIUM POLISHED', 'MEDIUM BRUSHED',
            'LARGE ANODIZED', 'LARGE BURNISHED', 'LARGE PLATED', 'LARGE POLISHED', 'LARGE BRUSHED',
            'ECONOMY ANODIZED', 'ECONOMY BURNISHED', 'ECONOMY PLATED', 'ECONOMY POLISHED', 'ECONOMY BRUSHED',
            'PROMO ANODIZED', 'PROMO BURNISHED', 'PROMO PLATED', 'PROMO POLISHED', 'PROMO BRUSHED'
        ]
        for _ in range(queries_per_type):
            type_start = random.choice(TYPE_CHOICES)
            queries.append(f"MATCH (p:Part) WHERE p.p_type STARTS WITH '{type_start.upper()}' RETURN p.p_name, p.p_type")

        # Custom Q8: Scan with Date Filter
        for _ in range(queries_per_type):
            year = random.randint(1996, 1998)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            queries.append(f"MATCH (o:Order) WHERE o.o_orderdate > date('{year}-{month:02d}-{day:02d}') RETURN o.o_orderkey, o.o_orderdate")

        # Custom Q9: Count Aggregation (All)
        for _ in range(queries_per_type):
            queries.append("MATCH (c:Customer) RETURN count(c) AS total_customers")

        # Custom Q10: Group-by Aggregation (Simple)
        for _ in range(queries_per_type):
            queries.append("MATCH (o:Order) RETURN o.o_orderstatus, count(o) AS order_count ORDER BY order_count DESC")

        # Custom Q11: Simple AVG/SUM Aggregation
        for _ in range(queries_per_type):
            queries.append("MATCH (li:LineItem) RETURN sum(li.l_quantity) AS total_qty, avg(li.l_extendedprice) AS avg_price, min(li.l_discount) AS min_discount")

        # Custom Q12: Simple DISTINCT
        for _ in range(queries_per_type):
            queries.append("MATCH (c:Customer) RETURN count(DISTINCT c.c_mktsegment) AS market_segments")

        # Custom Q13: Scan with AND
        for _ in range(queries_per_type):
            size = random.randint(10, 40)
            price = random.randint(1000, 1500)
            queries.append(f"MATCH (p:Part) WHERE p.p_size > {size} AND p.p_retailprice < {price} RETURN p.p_name, p.p_size, p.p_retailprice")

        # Custom Q14: Scan with OR
        for _ in range(queries_per_type):
            region1 = random.choice(REGION_CHOICES)
            region2 = random.choice(list(set(REGION_CHOICES) - {region1}))
            queries.append(f"MATCH (r:Region) WHERE r.r_name = '{region1}' OR r.r_name = '{region2}' RETURN r.r_name")

        # Custom Q15: Scan with NOT
        for _ in range(queries_per_type):
            status = random.choice(ORDER_STATUS_CHOICES)
            queries.append(f"MATCH (o:Order) WHERE NOT o.o_orderstatus = '{status}' RETURN o.o_orderkey, o.o_orderstatus LIMIT 100")

        # Custom Q16: 1-Hop Traversal (Find orders for a customer)
        for _ in range(queries_per_type):
            name = get_random_name('Customer', 30000)
            queries.append(f"""
MATCH (c:Customer)-[:PLACED]->(o:Order)
WHERE c.c_name = '{name}'
RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
""")

        # Custom Q17: 1-Hop with Filter and Aggregation (Count items in high-priority orders)
        for _ in range(queries_per_type):
            priority = random.choice(ORDER_PRIORITY_CHOICES[:2]) # '1-URGENT' or '2-HIGH'
            queries.append(f"""
MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
WHERE o.o_orderpriority = '{priority}'
RETURN o.o_orderkey, count(li) AS items
ORDER BY items DESC
LIMIT 50
""")

        # Custom Q18: 2-Hop Traversal (Find items for a customer)
        for _ in range(queries_per_type):
            name = get_random_name('Customer', 30000)
            queries.append(f"""
MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
WHERE c.c_name = '{name}'
RETURN c.c_name, count(li) AS total_items
""")

        # Custom Q19: 2-Hop with Aggregation (Count orders per nation)
        for _ in range(queries_per_type):
            nation = random.choice(NATIONS)
            queries.append(f"""
MATCH (n:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)
WHERE n.n_name = '{nation}'
RETURN n.n_name, count(o) AS orders_from_nation
""")

        # Custom Q20: 3-Hop Traversal (Find parts from a supplier)
        for _ in range(queries_per_type):
            name = get_random_name('Supplier', 2000)
            queries.append(f"""
MATCH (s:Supplier)<-[:SUPPLIED_BY]-(:PartSupp)-[:IS_FOR_PART]->(p:Part)
WHERE s.s_name = '{name}'
RETURN p.p_name, p.p_mfgr, p.p_retailprice
LIMIT 100
""")

        # Custom Q21: 3-Hop with Property Filter (Customers in a region)
        for _ in range(queries_per_type):
            region = random.choice(REGION_CHOICES)
            queries.append(f"""
MATCH (r:Region)<-[:IS_IN_REGION]-(n:Nation)<-[:IS_IN_NATION]-(c:Customer)
WHERE r.r_name = '{region}'
RETURN c.c_name, n.n_name
LIMIT 200
""")

        # Custom Q22: Complex Path (4-Hop) and Aggregation
        for _ in range(queries_per_type):
            key = random.randint(1, 15000) # Assuming 150k customers, SF=1
            queries.append(f"""
MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
WHERE c.c_custkey = {key}
RETURN p.p_name, count(p) AS part_count
ORDER BY part_count DESC
LIMIT 10
""")

        # Custom Q23: Multi-hop with CONTAINS (Find orders for a part type)
        for _ in range(queries_per_type):
            word = random.choice(P_NAME_WORDS)
            queries.append(f"""
MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(:LineItem)<-[:CONTAINS_ITEM]-(o:Order)
WHERE p.p_name CONTAINS '{word}'
RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
LIMIT 50
""")

        # Custom Q24: Aggregation on Traversal (Supplier stock value)
        for _ in range(queries_per_type):
            balance = random.randint(0, 1000)
            queries.append(f"""
MATCH (s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)
WHERE s.s_acctbal < {balance}
RETURN s.s_name, sum(ps.ps_supplycost * ps.ps_availqty) AS stock_value
ORDER BY stock_value DESC
LIMIT 20
""")

        # Custom Q25: Complex Path with Multiple Filters
        for _ in range(queries_per_type):
            nation = random.choice(NATIONS)
            price = random.randint(1500, 2000)
            qty = random.randint(5000, 8000)
            queries.append(f"""
MATCH (n:Nation {{n_name: '{nation}'}})<-[:IS_IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)-[:IS_FOR_PART]->(p:Part)
WHERE p.p_retailprice > {price} AND ps.ps_availqty > {qty}
RETURN s.s_name, p.p_name, ps.ps_supplycost, p.p_retailprice
LIMIT 50
""")


        print(f"Generated {len(queries)} query variants ({queries_per_type} per query type)")


        return queries[:num_queries]

    def get_plan(self, query: str) -> dict:
        """
        Get query execution plan using EXPLAIN (no execution).

        Args:
            query: Cypher query string

        Returns:
            Query plan as dictionary
        """
        with self.driver.session() as session:
            result = session.run(f"EXPLAIN {query}")
            summary = result.consume()
            plan = summary.plan
            return plan

    def execute_query(self, query: str, num_runs: int = 1, show_details: bool = False) -> float:
        """
        Execute a query multiple times and return average execution time.

        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging

        Returns:
            Average execution time in seconds
        """
        execution_times = []

        with self.driver.session() as session:
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(query)

                if show_details:
                    print(query)
                    print()
                    # print("Result sample:")
                    print(result.data())
                    print("-"*40)

                result.consume()  # Ensure full execution
                end_time = time.time()
                execution_times.append(end_time - start_time)

        return np.mean(execution_times)

    def get_plan_and_execute(self, query: str, num_runs: int = 1, show_details: bool = False) -> tuple[dict, float]:
        """
        Get query plan with EXPLAIN and measure actual execution time.

        Args:
            query: Cypher query string
            num_runs: Number of times to execute for averaging

        Returns:
            Tuple of (plan, average_execution_time_seconds)
        """
        plan = self.get_plan(query)

        # Measure actual execution time
        execution_time = self.execute_query(query, num_runs, show_details=show_details)

        return plan, execution_time

    def collect_workload(self, num_queries: int = 350,
                         num_runs_per_query: int = 1, show_details: bool = False) -> tuple[list[str], list[dict], list[float]]:
        """
        Collect a workload of query plans and execution times.

        Args:
            num_queries: Total number of queries to generate
            num_runs_per_query: Number of executions per query for averaging

        Returns:
            Tuple of (queries, plans, execution_times)
        """
        # Generate queries
        queries = self.generate_workload_queries(num_queries)

        print(f"\nExecuting {len(queries)} queries...")
        print(f"Each query will be executed {num_runs_per_query} times for averaging.\n")

        all_plans = []
        all_times = []

        for i, query in enumerate(queries):
            try:
                # Get plan and execution time
                plan, exec_time = self.get_plan_and_execute(query, num_runs_per_query, show_details=show_details)

                all_plans.append(plan)
                all_times.append(exec_time)

                if i % 100 == 0 and i > 0:
                    print(f"Extracted {i} / {len(queries)} plans...")

            except Exception as e:
                print(f" ERROR: {str(e)}")
                continue

        print(f"\n{'='*60}")
        print(f"Workload collection complete!")
        print(f"  Total queries: {len(queries)}")
        print(f"  Average execution time: {np.mean(all_times):.4f}s")
        print(f"  Min/Max execution time: {np.min(all_times):.4f}s / {np.max(all_times):.4f}s")
        print(f"{'='*60}\n")

        return queries, all_plans, all_times

    def get_plan_statistics(self, plans: list[dict]) -> dict[str, Any]:
        """
        Compute statistics about the collected plans.

        Args:
            plans: List of query plans

        Returns:
            Dictionary with statistics
        """
        operator_counts = {}
        total_operators = 0
        max_depth = 0

        def analyze_node(node, depth=0):
            nonlocal total_operators, max_depth

            total_operators += 1
            max_depth = max(max_depth, depth)

            op_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1

            for child in node.get('children', []):
                analyze_node(child, depth + 1)

        for plan in plans:
            max_depth = 0  # Reset for each plan
            analyze_node(plan)

        return {
            'total_operators': total_operators,
            'unique_operators': len(operator_counts),
            'operator_counts': operator_counts,
            'max_plan_depth': max_depth,
            'avg_operators_per_plan': total_operators / len(plans) if plans else 0
        }
