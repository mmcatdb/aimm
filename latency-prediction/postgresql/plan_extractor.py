import json
import time
import random
from typing import List, Dict, Any, Tuple
from config import DatabaseConfig


class PlanExtractor:
    """Extracts query plans and execution statistics from PostgreSQL."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        
    def execute_with_plan(self, query: str, clear_cache: bool = True) -> Tuple[Dict, float]:
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
            
            with conn.cursor() as cur:
                # Clear cache if requested (simulates cold cache)
                if clear_cache:
                    try:
                        cur.execute("DISCARD ALL;")
                    except Exception as e:
                        # If DISCARD ALL fails, try alternative cache clearing
                        print(f"Warning: Could not clear cache: {e}")
                        pass
                
                # Get the plan with execution statistics
                explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}"
                
                cur.execute(explain_query)
                result = cur.fetchone()
                
                # Parse JSON plan
                plan_json = result[0][0]  # EXPLAIN returns list of plans
                
                # Extract actual execution time from plan
                execution_time = plan_json['Execution Time']  # in ms
                
                return plan_json, execution_time
                
        finally:
            conn.close()
    
    def generate_tpch_queries(self, num_queries: int = 100, scale_factor: int = 1) -> List[str]:
        """
        Generate TPC-H style queries with parameter variations.
        
        Args:
            num_queries: Number of queries to generate
            scale_factor: TPC-H scale factor (affects data size)
            
        Returns:
            List of SQL query strings
        """
        queries = []

        NUM_QUERY_TYPES = 20
        
        # --- Define constant lists for parameters ---
        REGION_CHOICES = ['ASIA', 'AMERICA', 'EUROPE', 'MIDDLE EAST', 'AFRICA']
        SEGMENT_CHOICES = ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
        SHIPMODES_ALL = ['MAIL', 'SHIP', 'AIR', 'TRUCK', 'RAIL', 'FOB', 'REG AIR']

        NATIONS = ["ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA",  
                       "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ",  
                       "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU",
                       "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"]
        NATION_REGION_PAIRS = [
            ('ALGERIA', 'AFRICA'), ('ARGENTINA', 'AMERICA'), ('BRAZIL', 'AMERICA'),
            ('CANADA', 'AMERICA'), ('EGYPT', 'MIDDLE EAST'), ('ETHIOPIA', 'AFRICA'),
            ('FRANCE', 'EUROPE'), ('GERMANY', 'EUROPE'), ('INDIA', 'ASIA'),
            ('INDONESIA', 'ASIA'), ('IRAN', 'MIDDLE EAST'), ('IRAQ', 'MIDDLE EAST'),
            ('JAPAN', 'ASIA'), ('JORDAN', 'MIDDLE EAST'), ('KENYA', 'AFRICA'),
            ('MOROCCO', 'AFRICA'), ('MOZAMBIQUE', 'AFRICA'), ('PERU', 'AMERICA'),
            ('CHINA', 'ASIA'), ('ROMANIA', 'EUROPE'), ('SAUDI ARABIA', 'MIDDLE EAST'),
            ('VIETNAM', 'ASIA'), ('RUSSIA', 'EUROPE'), ('UNITED KINGDOM', 'EUROPE'),
            ('UNITED STATES', 'AMERICA')
        ]
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
        TYPE_CHOICES_Q2 = ['TIN', 'NICKEL', 'BRASS', 'STEEL', 'COPPER']
        NATION_CHOICES = [n[0] for n in NATION_REGION_PAIRS]
        WORD1_CHOICES = ['special', 'pending', 'unusual', 'express']
        WORD2_CHOICES = ['packages', 'requests', 'accounts', 'deposits']
        TYPE_CHOICES_Q16 = [
            'STANDARD ANODIZED', 'STANDARD BURNISHED', 'STANDARD PLATED', 'STANDARD POLISHED', 'STANDARD BRUSHED',
            'SMALL ANODIZED', 'SMALL BURNISHED', 'SMALL PLATED', 'SMALL POLISHED', 'SMALL BRUSHED',
            'MEDIUM ANODIZED', 'MEDIUM BURNISHED', 'MEDIUM PLATED', 'MEDIUM POLISHED', 'MEDIUM BRUSHED',
            'LARGE ANODIZED', 'LARGE BURNISHED', 'LARGE PLATED', 'LARGE POLISHED', 'LARGE BRUSHED',
            'ECONOMY ANODIZED', 'ECONOMY BURNISHED', 'ECONOMY PLATED', 'ECONOMY POLISHED', 'ECONOMY BRUSHED',
            'PROMO ANODIZED', 'PROMO BURNISHED', 'PROMO PLATED', 'PROMO POLISHED', 'PROMO BRUSHED'
        ]
        
        # TPC-H Query 1: Aggregate query with filtering
        # Tests: Sequential Scan, Aggregate, Sort
        for _ in range(num_queries // NUM_QUERY_TYPES):
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
        
        # TPC-H Query 2: Minimum Cost Supplier
        # Tests: Correlated subquery, joins, top-k
        for _ in range(num_queries // NUM_QUERY_TYPES):
            size = random.randint(1, 50)
            type_end = random.choice(TYPE_CHOICES_Q2)
            region_choice = random.choice(REGION_CHOICES)
            queries.append(f"""
                SELECT
                    s_acctbal,
                    s_name,
                    n_name,
                    p_partkey,
                    p_mfgr,
                    s_address,
                    s_phone,
                    s_comment
                FROM part, supplier, partsupp, nation, region
                WHERE
                    p_partkey = ps_partkey
                    AND s_suppkey = ps_suppkey
                    AND p_size = {size}
                    AND p_type LIKE '% {type_end}'
                    AND s_nationkey = n_nationkey
                    AND n_regionkey = r_regionkey
                    AND r_name = '{region_choice}'
                    AND ps_supplycost = (
                        SELECT MIN(ps_supplycost)
                        FROM partsupp, supplier, nation, region
                        WHERE
                            p_partkey = ps_partkey
                            AND s_suppkey = ps_suppkey
                            AND s_nationkey = n_nationkey
                            AND n_regionkey = r_regionkey
                            AND r_name = '{region_choice}'
                    )
                ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
                LIMIT 100
            """)
        
        # TPC-H Query 3: Join with aggregation and filtering
        # Tests: Hash Join, Aggregate, Index Scan, Sort
        for _ in range(num_queries // NUM_QUERY_TYPES):
            date_threshold = f"1995-03-{random.randint(1, 31):02d}"
            segment = random.choice(SEGMENT_CHOICES)
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
                    AND o_orderdate < date '{date_threshold}'
                    AND l_shipdate > date '{date_threshold}'
                GROUP BY l_orderkey, o_orderdate, o_shippriority
                ORDER BY revenue DESC, o_orderdate
                LIMIT 10
            """)
        
        # TPC-H Query 4: Exists subquery (tests semi-join or nested loop)
        # Tests: Hash Semi Join, Subquery execution
        for _ in range(num_queries // NUM_QUERY_TYPES):
            random_month_offset = random.randint(0, 57)    # Random month between Jan 1993 and Oct 1997
            year = 1993 + (random_month_offset // 12)
            month = 1 + (random_month_offset % 12)
            queries.append(f"""
                SELECT
                    o_orderpriority,
                    COUNT(*) as order_count
                FROM orders
                WHERE o_orderdate >= date '{year}-{month:02d}-01'
                    AND o_orderdate < date '{year}-{month:02d}-01' + interval '3 months'
                    AND EXISTS (
                        SELECT * FROM lineitem
                        WHERE l_orderkey = o_orderkey
                        AND l_commitdate < l_receiptdate
                    )
                GROUP BY o_orderpriority
                ORDER BY o_orderpriority
            """)

        # TPC-H Query 5: Multi-way join with aggregation
        # Tests: Multiple Hash Joins, Aggregate, Sort
        for _ in range(num_queries // NUM_QUERY_TYPES):
            year = random.randint(1993, 1997)
            region_choice = random.choice(REGION_CHOICES)
            queries.append(f"""
                SELECT
                    n_name,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue
                FROM customer, orders, lineitem, supplier, nation, region
                WHERE c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND l_suppkey = s_suppkey
                    AND c_nationkey = s_nationkey
                    AND s_nationkey = n_nationkey
                    AND n_regionkey = r_regionkey
                    AND r_name = '{region_choice}'
                    AND o_orderdate >= date '{year}-01-01'
                    AND o_orderdate < date '{year}-01-01' + interval '1 year'
                GROUP BY n_name
                ORDER BY revenue DESC
            """)
        
        # TPC-H Query 6: Simple aggregate with selective filter
        # Tests: Sequential Scan with Filter, Aggregate
        for _ in range(num_queries // NUM_QUERY_TYPES):
            year = random.randint(1993, 1997)
            discount_center = random.uniform(0.02, 0.09) # Generate center point
            discount_min = discount_center - 0.01
            discount_max = discount_center + 0.01
            quantity = random.randint(24, 25)
            queries.append(f"""
                SELECT
                    SUM(l_extendedprice * l_discount) as revenue
                FROM lineitem
                WHERE l_shipdate >= date '{year}-01-01'
                    AND l_shipdate < date '{year}-01-01' + interval '1 year'
                    AND l_discount BETWEEN {discount_min:.2f} AND {discount_max:.2f}
                    AND l_quantity < {quantity}
            """)
        
        # TPC-H Query 7: Complex multi-way join with union/aggregation
        # Tests: Hash Join, Aggregate, Sort, Multiple nation filtering
        for _ in range(num_queries // NUM_QUERY_TYPES):
            nation1 = random.choice(NATIONS)
            nation2 = random.choice(list(set(NATIONS) - {nation1}))
            
            queries.append(f"""
                SELECT
                    supp_nation,
                    cust_nation,
                    l_year,
                    SUM(volume) as revenue
                FROM (
                    SELECT
                        n1.n_name as supp_nation,
                        n2.n_name as cust_nation,
                        EXTRACT(year FROM l_shipdate) as l_year,
                        l_extendedprice * (1 - l_discount) as volume
                    FROM supplier, lineitem, orders, customer, nation n1, nation n2
                    WHERE s_suppkey = l_suppkey
                        AND o_orderkey = l_orderkey
                        AND c_custkey = o_custkey
                        AND s_nationkey = n1.n_nationkey
                        AND c_nationkey = n2.n_nationkey
                        AND ((n1.n_name = '{nation1}' AND n2.n_name = '{nation2}')
                            OR (n1.n_name = '{nation2}' AND n2.n_name = '{nation1}'))
                        AND l_shipdate BETWEEN date '1995-01-01' AND date '1996-12-31'
                ) as shipping
                GROUP BY supp_nation, cust_nation, l_year
                ORDER BY supp_nation, cust_nation, l_year
            """)
        
        # TPC-H Query 8: Market share analysis with nested aggregation
        # Tests: Complex joins, subquery, aggregate, group by with expressions
        SYLLABLE1 = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
        SYLLABLE2 = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
        SYLLABLE3 = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
        for _ in range(num_queries // NUM_QUERY_TYPES):
            nation, region = random.choice(NATION_REGION_PAIRS) 
            p_type = f"{random.choice(SYLLABLE1)} {random.choice(SYLLABLE2)} {random.choice(SYLLABLE3)}"
            queries.append(f"""
                SELECT
                    o_year,
                    SUM(CASE WHEN nation = '{nation}' THEN volume ELSE 0 END) / SUM(volume) as mkt_share
                FROM (
                    SELECT
                        EXTRACT(year FROM o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) as volume,
                        n2.n_name as nation
                    FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
                    WHERE p_partkey = l_partkey
                        AND s_suppkey = l_suppkey
                        AND l_orderkey = o_orderkey
                        AND o_custkey = c_custkey
                        AND c_nationkey = n1.n_nationkey
                        AND n1.n_regionkey = r_regionkey
                        AND r_name = '{region}'
                        AND s_nationkey = n2.n_nationkey
                        AND o_orderdate BETWEEN date '1995-01-01' AND date '1996-12-31'
                        AND p_type = '{p_type}'
                ) as all_nations
                GROUP BY o_year
                ORDER BY o_year
            """)
        
        # TPC-H Query 9: Product type profit measure
        # Tests: Multiple joins, aggregate with expressions, sort
        for _ in range(num_queries // NUM_QUERY_TYPES):
            color = random.choice(P_NAME_WORDS)
            queries.append(f"""
                SELECT
                    nation,
                    o_year,
                    SUM(amount) as sum_profit
                FROM (
                    SELECT
                        n_name as nation,
                        EXTRACT(year FROM o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                    FROM part, supplier, lineitem, partsupp, orders, nation
                    WHERE s_suppkey = l_suppkey
                        AND ps_suppkey = l_suppkey
                        AND ps_partkey = l_partkey
                        AND p_partkey = l_partkey
                        AND o_orderkey = l_orderkey
                        AND s_nationkey = n_nationkey
                        AND p_name LIKE '%{color}%'
                ) as profit
                GROUP BY nation, o_year
                ORDER BY nation, o_year DESC
            """)
        
        # TPC-H Query 10: Join with aggregation and top-k
        # Tests: Hash Join, Aggregate, Sort, Limit
        for _ in range(num_queries // NUM_QUERY_TYPES):
            random_month_offset = random.randint(0, 23)
            random_month_offset = random.randint(0, 23) # 24 possible months
            year = 1993 + ((random_month_offset + 1) // 12) # +1 for Feb start
            month = ((random_month_offset + 1) % 12) + 1 # +1 for 1-based index
            
            queries.append(f"""
                SELECT
                    c_custkey,
                    c_name,
                    SUM(l_extendedprice * (1 - l_discount)) as revenue,
                    c_acctbal,
                    n_name,
                    c_address,
                    c_phone,
                    c_comment
                FROM customer, orders, lineitem, nation
                WHERE c_custkey = o_custkey
                    AND l_orderkey = o_orderkey
                    AND o_orderdate >= date '{year}-{month:02d}-01'
                    AND o_orderdate < date '{year}-{month:02d}-01' + interval '3 months'
                    AND l_returnflag = 'R'
                    AND c_nationkey = n_nationkey
                GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
                ORDER BY revenue DESC
                LIMIT 20
            """)
        
        # TPC-H Query 11: Important Stock Identification
        # Tests: Aggregate with subquery in HAVING clause
        for _ in range(num_queries // NUM_QUERY_TYPES):
            nation_choice = random.choice(NATION_CHOICES)
            fraction = 0.0001 / max(1, scale_factor) # Avoid division by zero if SF=0
            queries.append(f"""
                SELECT
                    ps_partkey,
                    SUM(ps_supplycost * ps_availqty) as value
                FROM partsupp, supplier, nation
                WHERE
                    ps_suppkey = s_suppkey
                    AND s_nationkey = n_nationkey
                    AND n_name = '{nation_choice}'
                GROUP BY ps_partkey
                HAVING SUM(ps_supplycost * ps_availqty) > (
                    SELECT SUM(ps_supplycost * ps_availqty) * {fraction}
                    FROM partsupp, supplier, nation
                    WHERE
                        ps_suppkey = s_suppkey
                        AND s_nationkey = n_nationkey
                        AND n_name = '{nation_choice}'
                )
                ORDER BY value DESC
            """)
            
        # TPC-H Query 12: Join with conditional aggregation
        # Tests: Hash Join, Aggregate with CASE, Sort
        for _ in range(num_queries // NUM_QUERY_TYPES):
            year = random.randint(1993, 1997)
            shipmode1 = random.choice(SHIPMODES_ALL)
            shipmode2 = random.choice(list(set(SHIPMODES_ALL) - {shipmode1}))
            queries.append(f"""
                SELECT
                    l_shipmode,
                    SUM(CASE
                        WHEN o_orderpriority = '1-URGENT'
                            OR o_orderpriority = '2-HIGH'
                        THEN 1
                        ELSE 0
                    END) as high_priority,
                    SUM(CASE
                        WHEN o_orderpriority <> '1-URGENT'
                            AND o_orderpriority <> '2-HIGH'
                        THEN 1
                        ELSE 0
                    END) as low_priority
                FROM orders, lineitem
                WHERE o_orderkey = l_orderkey
                    AND l_shipmode IN ('{shipmode1}', '{shipmode2}')
                    AND l_commitdate < l_receiptdate
                    AND l_shipdate < l_commitdate
                    AND l_receiptdate >= date '{year}-01-01'
                    AND l_receiptdate < date '{year}-01-01' + interval '1 year'
                GROUP BY l_shipmode
                ORDER BY l_shipmode
            """)
        
        # TPC-H Query 13: Customer Distribution
        # Tests: Left Outer Join, Aggregate, Subquery
        for _ in range(num_queries // NUM_QUERY_TYPES):
            word1 = random.choice(WORD1_CHOICES)
            word2 = random.choice(WORD2_CHOICES)
            queries.append(f"""
                SELECT
                    c_count,
                    COUNT(*) as custdist
                FROM (
                    SELECT
                        c_custkey,
                        COUNT(o_orderkey)
                    FROM customer
                    LEFT OUTER JOIN orders ON
                        c_custkey = o_custkey
                        AND o_comment NOT LIKE '%{word1}%{word2}%'
                    GROUP BY c_custkey
                ) as c_orders (c_custkey, c_count)
                GROUP BY c_count
                ORDER BY custdist DESC, c_count DESC
            """)

        # TPC-H Query 14: Promotion effect
        # Tests: Join, aggregate with conditional, simple filter
        for _ in range(num_queries // NUM_QUERY_TYPES):
            year = random.randint(1993, 1997)
            month = random.randint(1, 12)
            queries.append(f"""
                SELECT
                    100.00 * SUM(CASE
                        WHEN p_type LIKE 'PROMO%'
                        THEN l_extendedprice * (1 - l_discount)
                        ELSE 0
                    END) / SUM(l_extendedprice * (1 - l_discount)) as promo_revenue
                FROM lineitem, part
                WHERE l_partkey = p_partkey
                    AND l_shipdate >= date '{year}-{month:02d}-01'
                    AND l_shipdate < date '{year}-{month:02d}-01' + interval '1 month'
            """)
        
        # Tests: Common Table Expression (WITH clause), Aggregate, Subquery
        for _ in range(num_queries // NUM_QUERY_TYPES):
            random_month_offset = random.randint(0, 57) # Jan 1993 to Oct 1997
            year = 1993 + (random_month_offset // 12)
            month = 1 + (random_month_offset % 12)
            start_date = f"{year}-{month:02d}-01"
            queries.append(f"""
                WITH revenue (supplier_no, total_revenue) AS (
                    SELECT
                        l_suppkey,
                        SUM(l_extendedprice * (1 - l_discount))
                    FROM lineitem
                    WHERE l_shipdate >= date '{start_date}'
                        AND l_shipdate < date '{start_date}' + interval '3 months'
                    GROUP BY l_suppkey
                )
                SELECT
                    s_suppkey,
                    s_name,
                    s_address,
                    s_phone,
                    total_revenue
                FROM supplier, revenue
                WHERE s_suppkey = supplier_no
                    AND total_revenue = (
                        SELECT MAX(total_revenue)
                        FROM revenue
                    )
                ORDER BY s_suppkey
            """)

        # TPC-H Query 16: Parts/Supplier Relationship
        # Tests: Join, Aggregate (COUNT DISTINCT), NOT IN Subquery
        for _ in range(num_queries // NUM_QUERY_TYPES):
            brand = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
            p_type = random.choice(TYPE_CHOICES_Q16)
            sizes = random.sample(range(1, 51), 8) # 8 distinct sizes
            size_list = ", ".join(map(str, sizes))
            queries.append(f"""
                SELECT
                    p_brand,
                    p_type,
                    p_size,
                    COUNT(DISTINCT ps_suppkey) as supplier_cnt
                FROM partsupp, part
                WHERE
                    p_partkey = ps_partkey
                    AND p_brand <> '{brand}'
                    AND p_type NOT LIKE '{p_type}%'
                    AND p_size IN ({size_list})
                    AND ps_suppkey NOT IN (
                        SELECT s_suppkey
                        FROM supplier
                        WHERE s_comment LIKE '%Customer%Complaints%'
                    )
                GROUP BY p_brand, p_type, p_size
                ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
            """)



        #!!!! TPC-H Query 17: Small-Quantity-Order Revenue
        # Tests: Join, Correlated subquery
        # for _ in range(num_queries // NUM_QUERY_TYPES):
        #     brand = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
        #     container = f"{random.choice(SYLLABLE1_Q17)} {random.choice(SYLLABLE2_Q17)}"
        #     queries.append(f"""
        #         SELECT
        #             SUM(l_extendedprice) / 7.0 as avg_yearly
        #         FROM lineitem, part
        #         WHERE
        #             p_partkey = l_partkey
        #             AND p_brand = '{brand}'
        #             AND p_container = '{container}'
        #             AND l_quantity < (
        #                 SELECT 0.2 * AVG(l_quantity)
        #                 FROM lineitem
        #                 WHERE l_partkey = p_partkey
        #             )
        #     """)

        # TPC-H Query 18: Large volume customer ranking
        # Tests: Join with subquery, aggregate, having clause, top-k
        for _ in range(num_queries // NUM_QUERY_TYPES):
            quantity_threshold = random.randint(312, 315)
            queries.append(f"""
                SELECT
                    c_name,
                    c_custkey,
                    o_orderkey,
                    o_orderdate,
                    o_totalprice,
                    SUM(l_quantity)
                FROM customer, orders, lineitem
                WHERE o_orderkey IN (
                    SELECT l_orderkey
                    FROM lineitem
                    GROUP BY l_orderkey
                    HAVING SUM(l_quantity) > {quantity_threshold}
                )
                AND c_custkey = o_custkey
                AND o_orderkey = l_orderkey
                GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
                ORDER BY o_totalprice DESC, o_orderdate
                LIMIT 100
            """)
        
        # TPC-H Query 19: Discounted revenue for specific products
        # Tests: Join with complex OR conditions, aggregate
        for _ in range(num_queries // NUM_QUERY_TYPES):
            quantity1 = random.randint(1, 10)    
            quantity2 = random.randint(10, 20)   
            quantity3 = random.randint(20, 30)   
            
            brand1 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}" 
            brand2 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
            brand3 = f"Brand#{random.randint(1, 5)}{random.randint(1, 5)}"
            
            queries.append(f"""
                SELECT
                    SUM(l_extendedprice * (1 - l_discount)) as revenue
                FROM lineitem, part
                WHERE (
                    p_partkey = l_partkey
                    AND p_brand = '{brand1}'
                    AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                    AND l_quantity >= {quantity1} AND l_quantity <= {quantity1 + 10}
                    AND p_size BETWEEN 1 AND 5
                    AND l_shipmode IN ('AIR', 'AIR REG')
                    AND l_shipinstruct = 'DELIVER IN PERSON'
                )
                OR (
                    p_partkey = l_partkey
                    AND p_brand = '{brand2}'
                    AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                    AND l_quantity >= {quantity2} AND l_quantity <= {quantity2 + 10}
                    AND p_size BETWEEN 1 AND 10
                    AND l_shipmode IN ('AIR', 'AIR REG')
                    AND l_shipinstruct = 'DELIVER IN PERSON'
                )
                OR (
                    p_partkey = l_partkey
                    AND p_brand = '{brand3}'
                    AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                    AND l_quantity >= {quantity3} AND l_quantity <= {quantity3 + 10}
                    AND p_size BETWEEN 1 AND 15
                    AND l_shipmode IN ('AIR', 'AIR REG')
                    AND l_shipinstruct = 'DELIVER IN PERSON'
                )
            """)
        
        #!!!!!!!!!! TPC-H Query 20: Potential Part Promotion
        # Tests: Multiple nested/correlated subqueries
        # for _ in range(num_queries // NUM_QUERY_TYPES):
        #     color = random.choice(P_NAME_WORDS)
        #     year = random.randint(1993, 1997)
        #     start_date = f"{year}-01-01"
        #     nation_choice = random.choice(NATION_CHOICES)
        #     queries.append(f"""
        #         SELECT
        #             s_name,
        #             s_address
        #         FROM supplier, nation
        #         WHERE s_suppkey IN (
        #             SELECT ps_suppkey
        #             FROM partsupp
        #             WHERE ps_partkey IN (
        #                 SELECT p_partkey
        #                 FROM part
        #                 WHERE p_name LIKE '{color}%'
        #             )
        #             AND ps_availqty > (
        #                 SELECT 0.5 * SUM(l_quantity)
        #                 FROM lineitem
        #                 WHERE l_partkey = ps_partkey
        #                     AND l_suppkey = ps_suppkey
        #                     AND l_shipdate >= date '{start_date}'
        #                     AND l_shipdate < date '{start_date}' + interval '1 year'
        #             )
        #         )
        #         AND s_nationkey = n_nationkey
        #         AND n_name = '{nation_choice}'
        #         ORDER BY s_name
        #     """)

        # TPC-H Query 21: Suppliers Who Kept Orders Waiting
        # Tests: EXISTS and NOT EXISTS subqueries
        for _ in range(num_queries // NUM_QUERY_TYPES):
            nation_choice = random.choice(NATION_CHOICES)
            queries.append(f"""
                SELECT
                    s_name,
                    COUNT(*) as numwait
                FROM supplier, lineitem l1, orders, nation
                WHERE
                    s_suppkey = l1.l_suppkey
                    AND o_orderkey = l1.l_orderkey
                    AND o_orderstatus = 'F'
                    AND l1.l_receiptdate > l1.l_commitdate
                    AND EXISTS (
                        SELECT *
                        FROM lineitem l2
                        WHERE l2.l_orderkey = l1.l_orderkey
                            AND l2.l_suppkey <> l1.l_suppkey
                    )
                    AND NOT EXISTS (
                        SELECT *
                        FROM lineitem l3
                        WHERE l3.l_orderkey = l1.l_orderkey
                            AND l3.l_suppkey <> l1.l_suppkey
                            AND l3.l_receiptdate > l3.l_commitdate
                    )
                    AND s_nationkey = n_nationkey
                    AND n_name = '{nation_choice}'
                GROUP BY s_name
                ORDER BY numwait DESC, s_name
                LIMIT 100
            """)
            
        # TPC-H Query 22: Global Sales Opportunity
        # Tests: Correlated NOT EXISTS, Substring, Aggregate with subquery
        for _ in range(num_queries // NUM_QUERY_TYPES):
            # Country codes are '10' through '34' (from 25 nations, index 0-24)
            country_codes = random.sample(range(10, 35), 7)
            country_code_list = ", ".join([f"'{c}'" for c in country_codes])
            queries.append(f"""
                SELECT
                    cntrycode,
                    COUNT(*) as numcust,
                    SUM(c_acctbal) as totacctbal
                FROM (
                    SELECT
                        SUBSTRING(c_phone FROM 1 FOR 2) as cntrycode,
                        c_acctbal
                    FROM customer
                    WHERE
                        SUBSTRING(c_phone FROM 1 FOR 2) IN ({country_code_list})
                        AND c_acctbal > (
                            SELECT AVG(c_acctbal)
                            FROM customer
                            WHERE c_acctbal > 0.00
                                AND SUBSTRING(c_phone FROM 1 FOR 2) IN ({country_code_list})
                        )
                        AND NOT EXISTS (
                            SELECT *
                            FROM orders
                            WHERE o_custkey = c_custkey
                        )
                ) as custsale
                GROUP BY cntrycode
                ORDER BY cntrycode
            """)

        return queries[:num_queries]
    
    def collect_training_data(self, num_queries: int = 1000, clear_cache: bool = True) -> List[Dict]:
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
        
        import time

        last = time.time()

        for i, query in enumerate(queries):
            if i % (num_queries / 20) == 0:
                print(f"Progress: {i}/{len(queries)} ({100*i//len(queries)}%) - time: {time.time()-last:.2f}s")
                last = time.time()
            
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