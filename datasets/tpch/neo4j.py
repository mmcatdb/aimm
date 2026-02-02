from typing_extensions import override
from datasets.database import Database
import datetime
import random

class TpchNeo4j(Database):
    NUM_QUERY_TYPES = 32 # Total number of different query types implemented

    @override
    def id(self) -> str:
        return 'tpch_neo4j'

    @override
    def _generate_train_queries(self, num_queries: int):
        # This will now be smaller, distributing the total num_queries over all 32 types
        queries_per_type = num_queries // TpchNeo4j.NUM_QUERY_TYPES

        # --- Define constant lists for parameters (from PostgreSQL version) ---
        SHIPMODES_ALL = ['MAIL', 'SHIP', 'AIR', 'TRUCK', 'RAIL', 'FOB', 'REG AIR']
        REGION_CHOICES = ['ASIA', 'AMERICA', 'EUROPE', 'MIDDLE EAST', 'AFRICA']
        SEGMENT_CHOICES = ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
        NATIONS = ['ALGERIA', 'ARGENTINA', 'BRAZIL', 'CANADA', 'EGYPT', 'ETHIOPIA', 'FRANCE', 'GERMANY', 'INDIA', 'INDONESIA', 'IRAN', 'IRAQ', 'JAPAN', 'JORDAN', 'KENYA', 'MOROCCO', 'MOZAMBIQUE', 'PERU', 'CHINA', 'ROMANIA', 'SAUDI ARABIA', 'VIETNAM', 'RUSSIA', 'UNITED KINGDOM', 'UNITED STATES']
        P_NAME_WORDS = [ 'almond', 'antique', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanched', 'blue', 'blush', 'brown', 'burlywood', 'burnished', 'chartreuse', 'chiffon', 'chocolate', 'coral', 'cornflower', 'cornsilk', 'cream', 'cyan', 'dark', 'deep', 'dim', 'dodger', 'drab', 'firebrick', 'floral', 'forest', 'frosted', 'gainsboro', 'ghost', 'goldenrod', 'green', 'grey', 'honeydew', 'hot', 'indian', 'ivory', 'khaki', 'lace', 'lavender', 'lawn', 'lemon', 'light', 'lime', 'linen', 'magenta', 'maroon', 'medium', 'metallic', 'midnight', 'mint', 'misty', 'moccasin', 'navajo', 'navy', 'olive', 'orange', 'orchid', 'pale', 'papaya', 'peach', 'peru', 'pink', 'plum', 'powder', 'puff', 'purple', 'red', 'rose', 'rosy', 'royal', 'saddle', 'salmon', 'sandy', 'seashell', 'sienna', 'sky', 'slate', 'smoke', 'snow', 'spring', 'steel', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'yellow' ]
        CONTAINER_CHOICES_SM = ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']
        CONTAINER_CHOICES_LG = ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
        ORDER_STATUS_CHOICES = ['F', 'O', 'P']
        ORDER_PRIORITY_CHOICES = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']

        # Q1: Pricing Summary Report
        for _ in range(queries_per_type):
            delta = random.randint(60, 120)
            base_date = datetime.date(1998, 12, 1)
            target_date = base_date - datetime.timedelta(days=delta)
            date_str = target_date.strftime('%Y-%m-%d')

            self._train_query(f'''
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
            ''')

        # Q5: Local Supplier Volume
        for _ in range(queries_per_type):
            region = random.choice(REGION_CHOICES)
            year = random.randint(1993, 1997)
            start_date = f'{year}-01-01'
            end_date = f'{year + 1}-01-01'

            self._train_query(f'''
                MATCH (r:Region {{r_name: '{region}'}})<-[:IS_IN_REGION]-(n:Nation)
                MATCH (n)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
                MATCH (n)<-[:IS_IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li)
                WHERE o.o_orderdate >= date('{start_date}')
                AND o.o_orderdate < date('{end_date}')
                WITH n.n_name AS nation_name,
                    sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
                RETURN nation_name, revenue
                ORDER BY revenue DESC
            ''')

        # Q6: Forecasting Revenue Change
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)

            discount_center = random.uniform(0.02, 0.09)
            discount_low = discount_center - 0.01
            discount_high = discount_center + 0.01

            quantity = random.randint(24, 25)

            self._train_query(f'''
                MATCH (li:LineItem)
                WHERE li.l_shipdate >= date('{year}-01-01')
                AND li.l_shipdate < date('{year + 1}-01-01')
                AND li.l_discount >= {discount_low:.2f}
                AND li.l_discount <= {discount_high:.2f}
                AND li.l_quantity < {quantity}
                RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
            ''')

        # Q10: Returned Item Reporting (Fix: Date generation)
        for _ in range(queries_per_type):
            random_month_offset = random.randint(0, 23)  # 24 possible months (Feb 1993 to Jan 1995)
            year = 1993 + ((random_month_offset + 1) // 12)
            month = ((random_month_offset + 1) % 12) + 1
            start_date = f'{year}-{month:02d}-01'

            # Add 3 months
            end_month = month + 3
            end_year = year
            if end_month > 12:
                end_month -= 12
                end_year += 1
            end_date = f'{end_year}-{end_month:02d}-01'

            self._train_query(f'''
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
            ''')

        # Q12: Shipping Modes and Order Priority (Fix: Shipmode generation)
        for _ in range(queries_per_type):
            shipmodes = random.sample(SHIPMODES_ALL, 2)
            year = random.randint(1993, 1997)

            self._train_query(f'''
                MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
                WHERE li.l_shipmode IN {shipmodes}
                AND li.l_commitdate < li.l_receiptdate
                AND li.l_shipdate < li.l_commitdate
                AND li.l_receiptdate >= date('{year}-01-01')
                AND li.l_receiptdate < date('{year + 1}-01-01')
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
            ''')

        # Q14: Promotion Effect (Fix: Month range and end date logic)
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            month = random.randint(1, 12)  # Fix: Was 1-11
            start_date = f'{year}-{month:02d}-01'

            end_month = month + 1
            end_year = year
            if end_month > 12:
                end_month = 1
                end_year += 1
            end_date = f'{end_year}-{end_month:02d}-01'

            self._train_query(f'''
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
            ''')

        # Q19: Discounted Revenue (Fix: Brand generation)
        for _ in range(queries_per_type):
            brand1 = f'Brand#{random.randint(1, 5)}{random.randint(1, 5)}'
            brand2 = f'Brand#{random.randint(1, 5)}{random.randint(1, 5)}'
            brand3 = f'Brand#{random.randint(1, 5)}{random.randint(1, 5)}'

            qty1 = random.randint(1, 10)
            qty2 = random.randint(10, 20)
            qty3 = random.randint(20, 30)

            self._train_query(f'''
                MATCH (li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
                WHERE li.l_shipinstruct = 'DELIVER IN PERSON'
                AND li.l_shipmode IN ['AIR', 'AIR REG']
                AND (
                    (
                    p.p_brand = '{brand1}'
                    AND p.p_container IN ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']
                    AND li.l_quantity >= {qty1} AND li.l_quantity <= {qty1 + 10}
                    AND p.p_size >= 1 AND p.p_size <= 5
                    ) OR (
                    p.p_brand = '{brand2}'
                    AND p.p_container IN ['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK']
                    AND li.l_quantity >= {qty2} AND li.l_quantity <= {qty2 + 10}
                    AND p.p_size >= 1 AND p.p_size <= 10
                    ) OR (
                    p.p_brand = '{brand3}'
                    AND p.p_container IN ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
                    AND li.l_quantity >= {qty3} AND li.l_quantity <= {qty3 + 10}
                    AND p.p_size >= 1 AND p.p_size <= 15
                    )
                )
                RETURN sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            ''')

        # Custom Q1: Simple Scan (All Nodes of a Label)
        for _ in range(queries_per_type):
            self._train_query('MATCH (n:Nation) RETURN n.n_name, n.n_comment LIMIT 100')

        # Custom Q2: Scan with Exact Property Match
        for _ in range(queries_per_type):
            name = TpchNeo4j.__get_random_name('Customer', 30000)
            self._train_query(f'MATCH (c:Customer {{c_name: \'{name}\'}}) RETURN c.c_address, c.c_phone')

        # Custom Q3: Scan with Numeric Filter
        for _ in range(queries_per_type):
            balance = random.randint(5000, 9500)
            self._train_query(f'MATCH (s:Supplier) WHERE s.s_acctbal > {balance} RETURN s.s_name, s.s_acctbal')

        # Custom Q4: Scan with Sort
        for _ in range(queries_per_type):
            self._train_query('MATCH (p:Part) RETURN p.p_name, p.p_retailprice ORDER BY p.p_retailprice DESC LIMIT 50')

        # Custom Q5: Scan with Sort + Limit
        for _ in range(queries_per_type):
            limit = random.randint(10, 50)
            self._train_query(f'MATCH (o:Order) RETURN o.o_orderkey, o.o_totalprice ORDER BY o.o_totalprice DESC LIMIT {limit}')

        # Custom Q6: Scan with IN list
        for _ in range(queries_per_type):
            containers = random.sample(CONTAINER_CHOICES_SM + CONTAINER_CHOICES_LG, 3)
            self._train_query(f'MATCH (p:Part) WHERE p.p_container IN {containers} RETURN p.p_name, p.p_container')

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
            self._train_query(f'MATCH (p:Part) WHERE p.p_type STARTS WITH \'{type_start.upper()}\' RETURN p.p_name, p.p_type')

        # Custom Q8: Scan with Date Filter
        for _ in range(queries_per_type):
            year = random.randint(1996, 1998)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            self._train_query(f'MATCH (o:Order) WHERE o.o_orderdate > date(\'{year}-{month:02d}-{day:02d}\') RETURN o.o_orderkey, o.o_orderdate')

        # Custom Q9: Count Aggregation (All)
        for _ in range(queries_per_type):
            self._train_query('MATCH (c:Customer) RETURN count(c) AS total_customers')

        # Custom Q10: Group-by Aggregation (Simple)
        for _ in range(queries_per_type):
            self._train_query('MATCH (o:Order) RETURN o.o_orderstatus, count(o) AS order_count ORDER BY order_count DESC')

        # Custom Q11: Simple AVG/SUM Aggregation
        for _ in range(queries_per_type):
            self._train_query('MATCH (li:LineItem) RETURN sum(li.l_quantity) AS total_qty, avg(li.l_extendedprice) AS avg_price, min(li.l_discount) AS min_discount')

        # Custom Q12: Simple DISTINCT
        for _ in range(queries_per_type):
            self._train_query('MATCH (c:Customer) RETURN count(DISTINCT c.c_mktsegment) AS market_segments')

        # Custom Q13: Scan with AND
        for _ in range(queries_per_type):
            size = random.randint(10, 40)
            price = random.randint(1000, 1500)
            self._train_query(f'MATCH (p:Part) WHERE p.p_size > {size} AND p.p_retailprice < {price} RETURN p.p_name, p.p_size, p.p_retailprice')

        # Custom Q14: Scan with OR
        for _ in range(queries_per_type):
            region1 = random.choice(REGION_CHOICES)
            region2 = random.choice(list(set(REGION_CHOICES) - {region1}))
            self._train_query(f'MATCH (r:Region) WHERE r.r_name = \'{region1}\' OR r.r_name = \'{region2}\' RETURN r.r_name')

        # Custom Q15: Scan with NOT
        for _ in range(queries_per_type):
            status = random.choice(ORDER_STATUS_CHOICES)
            self._train_query(f'MATCH (o:Order) WHERE NOT o.o_orderstatus = \'{status}\' RETURN o.o_orderkey, o.o_orderstatus LIMIT 100')

        # Custom Q16: 1-Hop Traversal (Find orders for a customer)
        for _ in range(queries_per_type):
            name = TpchNeo4j.__get_random_name('Customer', 30000)
            self._train_query(f'''
                MATCH (c:Customer)-[:PLACED]->(o:Order)
                WHERE c.c_name = '{name}'
                RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
            ''')

        # Custom Q17: 1-Hop with Filter and Aggregation (Count items in high-priority orders)
        for _ in range(queries_per_type):
            priority = random.choice(ORDER_PRIORITY_CHOICES[:2]) # '1-URGENT' or '2-HIGH'
            self._train_query(f'''
                MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
                WHERE o.o_orderpriority = '{priority}'
                RETURN o.o_orderkey, count(li) AS items
                ORDER BY items DESC
                LIMIT 50
            ''')

        # Custom Q18: 2-Hop Traversal (Find items for a customer)
        for _ in range(queries_per_type):
            name = TpchNeo4j.__get_random_name('Customer', 30000)
            self._train_query(f'''
                MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
                WHERE c.c_name = '{name}'
                RETURN c.c_name, count(li) AS total_items
            ''')

        # Custom Q19: 2-Hop with Aggregation (Count orders per nation)
        for _ in range(queries_per_type):
            nation = random.choice(NATIONS)
            self._train_query(f'''
                MATCH (n:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)
                WHERE n.n_name = '{nation}'
                RETURN n.n_name, count(o) AS orders_from_nation
            ''')

        # Custom Q20: 3-Hop Traversal (Find parts from a supplier)
        for _ in range(queries_per_type):
            name = TpchNeo4j.__get_random_name('Supplier', 2000)
            self._train_query(f'''
                MATCH (s:Supplier)<-[:SUPPLIED_BY]-(:PartSupp)-[:IS_FOR_PART]->(p:Part)
                WHERE s.s_name = '{name}'
                RETURN p.p_name, p.p_mfgr, p.p_retailprice
                LIMIT 100
            ''')

        # Custom Q21: 3-Hop with Property Filter (Customers in a region)
        for _ in range(queries_per_type):
            region = random.choice(REGION_CHOICES)
            self._train_query(f'''
                MATCH (r:Region)<-[:IS_IN_REGION]-(n:Nation)<-[:IS_IN_NATION]-(c:Customer)
                WHERE r.r_name = '{region}'
                RETURN c.c_name, n.n_name
                LIMIT 200
            ''')

        # Custom Q22: Complex Path (4-Hop) and Aggregation
        for _ in range(queries_per_type):
            key = random.randint(1, 15000) # Assuming 150k customers, SF=1
            self._train_query(f'''
                MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
                WHERE c.c_custkey = {key}
                RETURN p.p_name, count(p) AS part_count
                ORDER BY part_count DESC
                LIMIT 10
            ''')

        # Custom Q23: Multi-hop with CONTAINS (Find orders for a part type)
        for _ in range(queries_per_type):
            word = random.choice(P_NAME_WORDS)
            self._train_query(f'''
                MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(:LineItem)<-[:CONTAINS_ITEM]-(o:Order)
                WHERE p.p_name CONTAINS '{word}'
                RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
                LIMIT 50
            ''')

        # Custom Q24: Aggregation on Traversal (Supplier stock value)
        for _ in range(queries_per_type):
            balance = random.randint(0, 1000)
            self._train_query(f'''
                MATCH (s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)
                WHERE s.s_acctbal < {balance}
                RETURN s.s_name, sum(ps.ps_supplycost * ps.ps_availqty) AS stock_value
                ORDER BY stock_value DESC
                LIMIT 20
            ''')

        # Custom Q25: Complex Path with Multiple Filters
        for _ in range(queries_per_type):
            nation = random.choice(NATIONS)
            price = random.randint(1500, 2000)
            qty = random.randint(5000, 8000)
            self._train_query(f'''
                MATCH (n:Nation {{n_name: '{nation}'}})<-[:IS_IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)-[:IS_FOR_PART]->(p:Part)
                WHERE p.p_retailprice > {price} AND ps.ps_availqty > {qty}
                RETURN s.s_name, p.p_name, ps.ps_supplycost, p.p_retailprice
                LIMIT 50
            ''')

    @staticmethod
    def __get_random_name(table_name: str, max_id: int, min_id = 1, id_length = 9) -> str:
        num = random.randint(min_id, max_id)
        num_padding_zeroes = id_length - len(str(num))
        return f'{table_name}#{"0" * num_padding_zeroes}{num}'

    @override
    def _generate_test_queries(self):
        # Q1 variants (Original)
        self._test_query('Q1-Test-1', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate <= date('1998-10-15')
            WITH li.l_returnflag AS returnflag, li.l_linestatus AS linestatus, li
            RETURN
            returnflag,
            linestatus,
            sum(li.l_quantity) AS sum_qty,
            sum(li.l_extendedprice) AS sum_base_price
            ORDER BY returnflag, linestatus
            LIMIT 5
        ''')

        # Q5 variant (Original)
        self._test_query('Q5-Test-1', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
                (li)-[:IS_PRODUCT_SUPPLY]->(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier),
                (c)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
            WHERE r.r_name = 'ASIA'
            AND o.o_orderdate >= date('1994-01-01')
            AND o.o_orderdate < date('1995-01-01')
            AND s.s_nationkey = n.n_nationkey
            WITH n.n_name AS nation, li
            RETURN nation, sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            ORDER BY revenue DESC
            LIMIT 3
        ''')

        # Q6 variant (Original)
        self._test_query('Q6-Test-1', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1994-01-01')
            AND li.l_shipdate < date('1995-01-01')
            AND li.l_discount >= 0.05
            AND li.l_discount <= 0.07
            AND li.l_quantity < 24
            RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
        ''')

        # Q10 variant (Original)
        self._test_query('Q10-Test-1', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
                (c)-[:IS_IN_NATION]->(n:Nation)
            WHERE o.o_orderdate >= date('1993-07-01')
            AND o.o_orderdate < date('1993-10-01')
            AND li.l_returnflag = 'R'
            WITH c, n, li
            RETURN
            c.c_custkey,
            c.c_name,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue,
            c.c_acctbal,
            n.n_name
            ORDER BY revenue DESC
            LIMIT 5
        ''')

        # Simple aggregation query (Original)
        self._test_query('Simple-Agg-1', '''
            MATCH (li:LineItem)
            WHERE li.l_quantity > 30
            RETURN count(li) AS count, avg(li.l_extendedprice) AS avg_price
        ''')

        # Simple scan with limit (Original)
        self._test_query('Simple-Scan-1', '''
            MATCH (c:Customer)
            WHERE c.c_acctbal > 5000
            RETURN c.c_name, c.c_acctbal
            ORDER BY c.c_acctbal DESC
            LIMIT 10
        ''')

        # ========================================================================
        # CATEGORY 1: Simple Aggregation Queries
        # ========================================================================

        self._test_query('Simple Agg 1: Lineitem Summary', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate <= date('1998-08-01')
            RETURN
                li.l_returnflag AS returnflag,
                count(li) AS count,
                avg(li.l_quantity) AS avg_qty,
                sum(li.l_extendedprice) AS total_price
            ORDER BY returnflag
        ''')

        self._test_query('Simple Agg 2: Order Statistics', '''
            MATCH (o:Order)
            WHERE o.o_orderdate >= date('1996-01-01')
            RETURN
                o.o_orderpriority AS orderpriority,
                count(o) AS order_count,
                avg(o.o_totalprice) AS avg_price,
                min(o.o_totalprice) AS min_price,
                max(o.o_totalprice) AS max_price
            ORDER BY orderpriority
        ''')

        self._test_query('Simple Agg 3: Customer Segments', '''
            MATCH (c:Customer)
            WHERE c.c_acctbal > 0
            WITH
                c.c_mktsegment AS mktsegment,
                count(c) AS customer_count,
                avg(c.c_acctbal) AS avg_balance,
                sum(c.c_acctbal) AS total_balance
            RETURN
                mktsegment,
                customer_count,
                avg_balance,
                total_balance
            ORDER BY customer_count DESC
        ''')

        self._test_query('Simple Agg 4: Part Analysis', '''
            MATCH (p:Part)
            WHERE p.p_size >= 10 AND p.p_size <= 30
            WITH
                p.p_brand AS brand,
                p.p_type AS type,
                count(p) AS part_count,
                avg(p.p_retailprice) AS avg_price
            WHERE part_count > 5
            RETURN
                brand,
                type,
                part_count,
                avg_price
            ORDER BY brand, type
        ''')

        self._test_query('Simple Agg 5: Supplier Stats', '''
            MATCH (s:Supplier)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
            WHERE s.s_acctbal > 1000
            WITH
                r.r_name AS region_name,
                n.n_name AS nation_name,
                count(s) AS supplier_count,
                avg(s.s_acctbal) AS avg_balance,
                sum(s.s_acctbal) AS total_balance
            RETURN
                region_name,
                nation_name,
                supplier_count,
                avg_balance,
                total_balance
            ORDER BY supplier_count DESC
        ''')

        self._test_query('Simple Agg 6: Discount Analysis', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1997-01-01') AND li.l_shipdate < date('1998-01-01')
            RETURN
                li.l_linestatus AS linestatus,
                avg(li.l_discount) AS avg_discount,
                avg(li.l_tax) AS avg_tax,
                count(li) AS line_count
            ORDER BY linestatus
        ''')

        # ========================================================================
        # CATEGORY 2: Simple Join Queries
        # ========================================================================

        self._test_query('Join 1: Customer Orders', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)
            WHERE o.o_orderdate >= date('1995-01-01')
            WITH
                c.c_name AS name,
                c.c_mktsegment AS mktsegment,
                count(o) AS order_count,
                sum(o.o_totalprice) AS total_spent
            RETURN
                name,
                mktsegment,
                order_count,
                total_spent
            ORDER BY total_spent DESC
            LIMIT 100
        ''')

        self._test_query('Join 2: Parts and Suppliers', '''
            MATCH (ps:PartSupp)-[:IS_FOR_PART]->(p:Part)
            WHERE p.p_size > 20 AND ps.ps_supplycost < 100
            RETURN
                p.p_partkey AS partkey,
                p.p_name AS name,
                ps.ps_supplycost AS supplycost,
                ps.ps_availqty AS availqty
            ORDER BY supplycost
            LIMIT 200
        ''')

        self._test_query('Join 3: Order Details', '''
            MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
            WHERE o.o_orderdate >= date('1996-01-01') AND o.o_orderdate <= date('1996-03-31')
            AND li.l_quantity > 30
            RETURN
                o.o_orderkey AS orderkey,
                o.o_orderdate AS orderdate,
                li.l_linenumber AS linenumber,
                li.l_quantity AS quantity,
                li.l_extendedprice AS extendedprice
            ORDER BY orderdate, orderkey
            LIMIT 500
        ''')

        self._test_query('Join 4: Supplier Orders', '''
            MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)
            WHERE li.l_shipdate >= date('1996-01-01') AND li.l_shipdate < date('1997-01-01')
            WITH
                s.s_name AS name,
                s.s_address AS address,
                count(DISTINCT o) AS order_count,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            RETURN
                name,
                address,
                order_count,
                revenue
            ORDER BY revenue DESC
            LIMIT 50
        ''')

        self._test_query('Join 5: Customer Nation Analysis', '''
            MATCH (r:Region)<-[:IS_IN_REGION]-(n:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.o_orderdate >= date('1997-01-01')
            WITH
                r.r_name AS region_name,
                n.n_name AS nation_name,
                count(DISTINCT c) AS customer_count,
                count(o) AS total_orders,
                avg(o.o_totalprice) AS avg_order_value
            RETURN
                region_name,
                nation_name,
                customer_count,
                total_orders,
                avg_order_value
            ORDER BY total_orders DESC
        ''')

        self._test_query('Join 6: Part Lineitem Summary', '''
            MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li:LineItem)
            WHERE li.l_shipdate >= date('1995-01-01') AND p.p_size < 15
            WITH
                p.p_brand AS brand,
                p.p_container AS container,
                count(li) AS shipment_count,
                avg(li.l_quantity) AS avg_quantity,
                sum(li.l_extendedprice) AS total_value
            WHERE shipment_count > 10
            RETURN
                brand,
                container,
                shipment_count,
                avg_quantity,
                total_value
            ORDER BY brand, container
        ''')

        # ========================================================================
        # CATEGORY 3: Complex Multi-table Joins
        # ========================================================================

        self._test_query('Complex Join 1: Customer Segment Revenue', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
            WHERE li.l_shipdate >= date('1995-06-01')
            AND li.l_shipdate < date('1995-09-01')
            AND c.c_mktsegment = 'BUILDING'
            RETURN
                c.c_mktsegment AS mktsegment,
                avg(li.l_extendedprice * (1 - li.l_discount)) AS avg_revenue,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
                count(DISTINCT c) AS customer_count
        ''')

        self._test_query('Complex Join 2: Supplier Revenue Analysis', '''
            MATCH (r:Region)<-[:IS_IN_REGION]-(cn:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
                (li)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)-[:IS_IN_NATION]->(sn:Nation)-[:IS_IN_REGION]->(r)
            WHERE o.o_orderdate >= date('1994-01-01')
            AND o.o_orderdate < date('1995-01-01')
            WITH
                r.r_name AS region_name,
                sn.n_name AS supplier_nation,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            RETURN
                region_name,
                supplier_nation,
                revenue
            ORDER BY revenue DESC
            LIMIT 100
        ''')

        self._test_query('Complex Join 3: Part Supplier Customer Chain', '''
            MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li:LineItem)<-[:CONTAINS_ITEM]-(o:Order)<-[:PLACED]-(c:Customer)
            WHERE p.p_type CONTAINS 'BRASS'
            AND o.o_orderdate >= date('1996-01-01')
            AND o.o_orderdate < date('1997-01-01')
            WITH
                p.p_brand AS brand,
                c.c_mktsegment AS mktsegment,
                count(DISTINCT o) AS order_count,
                sum(li.l_quantity) AS total_quantity,
                avg(li.l_discount) AS avg_discount
            RETURN
                brand,
                mktsegment,
                order_count,
                total_quantity,
                avg_discount
            ORDER BY order_count DESC
        ''')

        self._test_query('Complex Join 4: Multi-way with Partsupp', '''
            MATCH (p:Part)<-[:IS_FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier),
                (li:LineItem)-[:IS_PRODUCT_SUPPLY]->(ps)
            WHERE li.l_shipdate >= date('1996-01-01')
            AND li.l_shipdate < date('1996-06-01')
            AND p.p_size > 15
            WITH
                p.p_partkey AS partkey,
                s.s_name AS name,
                ps.ps_supplycost AS supplycost,
                sum(li.l_quantity) AS total_shipped
            WHERE total_shipped > 50
            RETURN
                partkey,
                name,
                supplycost,
                total_shipped
            ORDER BY partkey, name
            LIMIT 100
        ''')

        self._test_query('Complex Join 5: Full Chain Analysis', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
            WHERE o.o_orderdate >= date('1997-01-01')
            AND o.o_orderdate < date('1997-07-01')
            AND li.l_discount > 0.05
            WITH
                c.c_mktsegment AS mktsegment,
                p.p_brand AS brand,
                count(li) AS transaction_count,
                avg(li.l_extendedprice * (1 - li.l_discount)) AS avg_net_price
            WHERE transaction_count > 5
            RETURN
                mktsegment,
                brand,
                transaction_count,
                avg_net_price
            ORDER BY mktsegment, brand
        ''')

        self._test_query('Complex Join 6: Regional Supply Chain', '''
            MATCH (r:Region)<-[:IS_IN_REGION]-(cn:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
                (li)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)-[:IS_IN_NATION]->(sn:Nation)-[:IS_IN_REGION]->(r)
            WHERE o.o_orderdate >= date('1995-01-01')
            AND o.o_orderdate < date('1996-01-01')
            WITH
                sn.n_name AS supplier_nation,
                cn.n_name AS customer_nation,
                r.r_name AS region_name,
                count(DISTINCT o) AS orders,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            WHERE orders > 10
            RETURN
                supplier_nation,
                customer_nation,
                region_name,
                orders,
                revenue
            ORDER BY revenue DESC
            LIMIT 50
        ''')

        # ========================================================================
        # CATEGORY 4: Selective Scans with Filters
        # ========================================================================

        self._test_query('Selective 1: Discount Range', '''
            MATCH (li:LineItem)
            WHERE li.l_discount >= 0.05 AND li.l_discount <= 0.07
            AND li.l_quantity < 24
            AND li.l_shipdate >= date('1994-01-01')
            WITH li, (li.l_extendedprice * li.l_discount) AS discount_revenue
            RETURN
                li.l_orderkey AS orderkey,
                li.l_linenumber AS linenumber,
                li.l_quantity AS quantity,
                li.l_extendedprice AS extendedprice,
                li.l_discount AS discount,
                discount_revenue
            ORDER BY discount_revenue DESC
            LIMIT 100
        ''')

        self._test_query('Selective 2: High Value Orders', '''
            MATCH (o:Order)
            WHERE o.o_totalprice > 300000
            AND o.o_orderdate >= date('1995-01-01')
            AND o.o_orderdate < date('1997-01-01')
            RETURN
                o.o_orderkey AS orderkey,
                o.o_custkey AS custkey,
                o.o_totalprice AS totalprice,
                o.o_orderdate AS orderdate
            ORDER BY totalprice DESC
            LIMIT 50
        ''')

        self._test_query('Selective 3: Premium Customers', '''
            MATCH (c:Customer)
            WHERE c.c_acctbal > 8000
            AND c.c_mktsegment IN ['AUTOMOBILE', 'MACHINERY']
            RETURN
                c.c_custkey AS custkey,
                c.c_name AS name,
                c.c_acctbal AS acctbal,
                c.c_mktsegment AS mktsegment
            ORDER BY acctbal DESC
            LIMIT 100
        ''')

        self._test_query('Selective 4: Specific Part Types', '''
            MATCH (p:Part)
            WHERE p.p_brand = 'Brand#23'
            AND p.p_container IN ['SM BOX', 'SM PACK']
            AND p.p_size >= 5 AND p.p_size <= 15
            RETURN
                p.p_partkey AS partkey,
                p.p_name AS name,
                p.p_brand AS brand,
                p.p_retailprice AS retailprice
            ORDER BY retailprice DESC
        ''')

        self._test_query('Selective 5: Late Shipments', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate > li.l_commitdate
            AND li.l_receiptdate >= date('1996-01-01')
            AND li.l_receiptdate < date('1996-06-01')
            RETURN
                li.l_orderkey AS orderkey,
                li.l_linenumber AS linenumber,
                li.l_shipdate AS shipdate,
                li.l_commitdate AS commitdate,
                li.l_receiptdate AS receiptdate
            ORDER BY shipdate
            LIMIT 200
        ''')

        self._test_query('Selective 6: Low Supply Cost', '''
            MATCH (ps:PartSupp)
            WHERE ps.ps_supplycost < 50
            AND ps.ps_availqty > 5000
            RETURN
                ps.ps_partkey AS partkey,
                ps.ps_suppkey AS suppkey,
                ps.ps_supplycost AS supplycost,
                ps.ps_availqty AS availqty
            ORDER BY supplycost, availqty DESC
            LIMIT 150
        ''')

        # ========================================================================
        # CATEGORY 5: Large Scans with Sorting
        # ========================================================================

        self._test_query('Large Scan 1: Sorted Lineitem by Price', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1997-01-01')
            AND li.l_shipdate < date('1997-04-01')
            RETURN
                li.l_orderkey AS orderkey,
                li.l_partkey AS partkey,
                li.l_suppkey AS suppkey,
                li.l_quantity AS quantity,
                li.l_extendedprice AS extendedprice
            ORDER BY extendedprice DESC
            LIMIT 200
        ''')

        self._test_query('Large Scan 2: Orders by Date', '''
            MATCH (o:Order)
            WHERE o.o_orderdate >= date('1996-01-01')
            AND o.o_orderdate < date('1997-01-01')
            RETURN
                o.o_orderkey AS orderkey,
                o.o_custkey AS custkey,
                o.o_totalprice AS totalprice,
                o.o_orderdate AS orderdate,
                o.o_orderpriority AS orderpriority
            ORDER BY orderdate DESC, totalprice DESC
            LIMIT 500
        ''')

        self._test_query('Large Scan 3: Parts by Price', '''
            MATCH (p:Part)
            WHERE p.p_retailprice > 1000
            RETURN
                p.p_partkey AS partkey,
                p.p_name AS name,
                p.p_brand AS brand,
                p.p_type AS type,
                p.p_retailprice AS retailprice
            ORDER BY retailprice DESC, partkey
            LIMIT 300
        ''')

        self._test_query('Large Scan 4: Customer Balance Ranking', '''
            MATCH (c:Customer)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
            WHERE c.c_acctbal > 0
            RETURN
                c.c_custkey AS custkey,
                c.c_name AS name,
                c.c_acctbal AS acctbal,
                c.c_mktsegment AS mktsegment,
                n.n_name AS nation_name,
                r.r_name AS region_name
            ORDER BY acctbal DESC, name
            LIMIT 400
        ''')

        self._test_query('Large Scan 5: Lineitem Quantity Sort', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1996-06-01')
            AND li.l_shipdate < date('1996-09-01')
            AND li.l_quantity > 40
            RETURN
                li.l_orderkey AS orderkey,
                li.l_partkey AS partkey,
                li.l_quantity AS quantity,
                li.l_extendedprice AS extendedprice,
                li.l_discount AS discount
            ORDER BY quantity DESC, extendedprice DESC
            LIMIT 250
        ''')

        self._test_query('Large Scan 6: Recent Shipments', '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1998-06-01')
            WITH li, (li.l_extendedprice * (1 - li.l_discount)) as net_price
            RETURN
                li.l_orderkey AS orderkey,
                li.l_linenumber AS linenumber,
                li.l_shipdate AS shipdate,
                li.l_receiptdate AS receiptdate,
                net_price
            ORDER BY shipdate DESC, net_price DESC
            LIMIT 300
        ''')

        # ========================================================================
        # CATEGORY 6: Aggregation with HAVING
        # ========================================================================

        self._test_query('Having 1: Large Order Aggregates', '''
            MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
            WITH
                o.o_orderkey AS orderkey,
                sum(li.l_quantity) AS total_qty,
                sum(li.l_extendedprice) AS total_price,
                count(li) AS line_count
            WHERE total_qty > 300
            RETURN
                orderkey,
                total_qty,
                total_price,
                line_count
            ORDER BY total_price DESC
            LIMIT 100
        ''')

        self._test_query('Having 2: High Volume Customers', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)
            WHERE o.o_orderdate >= date('1996-01-01')
            WITH
                c.c_custkey AS custkey,
                count(o) AS order_count,
                sum(o.o_totalprice) AS total_spent,
                avg(o.o_totalprice) AS avg_order_value
            WHERE order_count > 15
            RETURN
                custkey,
                order_count,
                total_spent,
                avg_order_value
            ORDER BY total_spent DESC
            LIMIT 50
        ''')

        self._test_query('Having 3: Popular Parts by Brand', '''
            MATCH (p:Part)
            WITH
                p.p_brand AS brand,
                count(p) AS part_count,
                avg(p.p_retailprice) AS avg_price,
                min(p.p_retailprice) AS min_price,
                max(p.p_retailprice) AS max_price
            WHERE part_count > 50
            RETURN
                brand,
                part_count,
                avg_price,
                min_price,
                max_price
            ORDER BY avg_price DESC
        ''')

        self._test_query('Having 4: High Revenue Suppliers', '''
            MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)
            WHERE li.l_shipdate >= date('1997-01-01')
            WITH
                s.s_suppkey AS suppkey,
                count(DISTINCT o) AS order_count,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
                avg(li.l_quantity) AS avg_quantity
            WHERE total_revenue > 500000
            RETURN
                suppkey,
                order_count,
                total_revenue,
                avg_quantity
            ORDER BY total_revenue DESC
            LIMIT 75
        ''')

        self._test_query('Having 5: Part Categories with High Sales', '''
            MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
            WHERE li.l_shipdate >= date('1996-01-01')
            AND li.l_shipdate < date('1997-01-01')
            WITH
                p.p_type AS type,
                count(DISTINCT o) AS order_count,
                sum(li.l_quantity) AS total_quantity,
                avg(li.l_extendedprice) AS avg_price
            WHERE total_quantity > 1000
            RETURN
                type,
                order_count,
                total_quantity,
                avg_price
            ORDER BY total_quantity DESC
            LIMIT 25
        ''')

        self._test_query('Having 6: Customer Segments with Volume', '''
            MATCH (c:Customer)-[:PLACED]->(o:Order)
            WHERE o.o_orderdate >= date('1997-01-01')
            WITH
                c.c_mktsegment AS mktsegment,
                count(DISTINCT c) AS customer_count,
                count(o) AS total_orders,
                avg(o.o_totalprice) AS avg_order_price
            WHERE total_orders > 1000
            RETURN
                mktsegment,
                customer_count,
                total_orders,
                avg_order_price
            ORDER BY total_orders DESC
        ''')
