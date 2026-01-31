import random
from datasets.database import Database

class TpchPostgres(Database):
    NUM_QUERY_TYPES = 6 # Total number of different query types implemented

    def __generate_train_queries(self, num_queries: int):
        queries_per_type = num_queries // TpchPostgres.NUM_QUERY_TYPES

        # TPC-H Query 1 variations
        for _ in range(queries_per_type):
            delta = random.randint(60, 120)
            self.__train_query(f'''
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
            ''')

        # TPC-H Query 3 variations
        for _ in range(queries_per_type):
            segment = random.choice(['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'])
            date = f'1995-03-{random.randint(1, 31):02d}'
            self.__train_query(f'''
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
            ''')

        # TPC-H Query 5 variations
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            self.__train_query(f'''
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
            ''')

        # TPC-H Query 6 variations
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            discount = random.uniform(0.02, 0.09)
            quantity = random.randint(20, 30)
            self.__train_query(f'''
                SELECT
                    SUM(l_extendedprice * l_discount) as revenue
                FROM lineitem
                WHERE l_shipdate >= date '{year}-01-01'
                    AND l_shipdate < date '{year}-01-01' + interval '1 year'
                    AND l_discount BETWEEN {discount} - 0.01 AND {discount} + 0.01
                    AND l_quantity < {quantity}
            ''')

        # TPC-H Query 10 variations
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            month = random.randint(1, 12)
            self.__train_query(f'''
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
            ''')

        # TPC-H Query 12 variations
        for _ in range(queries_per_type):
            year = random.randint(1993, 1997)
            mode1 = random.choice(['MAIL', 'SHIP', 'AIR', 'TRUCK'])
            mode2 = random.choice(['RAIL', 'FOB', 'REG AIR'])
            self.__train_query(f'''
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
            ''')

    def __generate_test_queries(self):
        # ========================================================================
        # CATEGORY 1: Simple Aggregation Queries
        # ========================================================================

        self.__test_query('Simple Agg 1: Lineitem Summary', '''
            SELECT
                l_returnflag,
                COUNT(*) as count,
                AVG(l_quantity) as avg_qty,
                SUM(l_extendedprice) as total_price
            FROM lineitem
            WHERE l_shipdate <= date '1998-08-01'
            GROUP BY l_returnflag
        ''')

        self.__test_query('Simple Agg 2: Order Statistics', '''
            SELECT
                o_orderpriority,
                COUNT(*) as order_count,
                AVG(o_totalprice) as avg_price,
                MIN(o_totalprice) as min_price,
                MAX(o_totalprice) as max_price
            FROM orders
            WHERE o_orderdate >= date '1996-01-01'
            GROUP BY o_orderpriority
        ''')

        self.__test_query('Simple Agg 3: Customer Segments', '''
            SELECT
                c_mktsegment,
                COUNT(*) as customer_count,
                AVG(c_acctbal) as avg_balance,
                SUM(c_acctbal) as total_balance
            FROM customer
            WHERE c_acctbal > 0
            GROUP BY c_mktsegment
            ORDER BY customer_count DESC
        ''')

        self.__test_query('Simple Agg 4: Part Analysis', '''
            SELECT
                p_brand,
                p_type,
                COUNT(*) as part_count,
                AVG(p_retailprice) as avg_price
            FROM part
            WHERE p_size BETWEEN 10 AND 30
            GROUP BY p_brand, p_type
            HAVING COUNT(*) > 5
        ''')

        self.__test_query('Simple Agg 5: Supplier Stats', '''
            SELECT
                r.r_name AS region_name,
                n.n_name AS nation_name,
                COUNT(*) AS supplier_count,
                AVG(s.s_acctbal) AS avg_balance,
                SUM(s.s_acctbal) AS total_balance
            FROM supplier AS s
            JOIN nation AS n ON s.s_nationkey = n.n_nationkey
            JOIN region AS r ON n.n_regionkey = r.r_regionkey
            WHERE s.s_acctbal > 1000
            GROUP BY r.r_name, n.n_name
            ORDER BY supplier_count DESC
        ''')

        self.__test_query('Simple Agg 6: Discount Analysis', '''
            SELECT
                l_linestatus,
                AVG(l_discount) as avg_discount,
                AVG(l_tax) as avg_tax,
                COUNT(*) as line_count
            FROM lineitem
            WHERE l_shipdate >= date '1997-01-01'
                AND l_shipdate < date '1998-01-01'
            GROUP BY l_linestatus
        ''')

        # ========================================================================
        # CATEGORY 2: Simple Join Queries
        # ========================================================================

        self.__test_query('Join 1: Customer Orders', '''
            SELECT
                c_name,
                c_mktsegment,
                COUNT(o_orderkey) as order_count,
                SUM(o_totalprice) as total_spent
            FROM customer
            JOIN orders ON c_custkey = o_custkey
            WHERE o_orderdate >= date '1995-01-01'
            GROUP BY c_custkey, c_name, c_mktsegment
            ORDER BY total_spent DESC
            LIMIT 100
        ''')

        self.__test_query('Join 2: Parts and Suppliers', '''
            SELECT
                p_partkey,
                p_name,
                ps_supplycost,
                ps_availqty
            FROM part
            JOIN partsupp ON p_partkey = ps_partkey
            WHERE p_size > 20
                AND ps_supplycost < 100
            ORDER BY ps_supplycost
            LIMIT 200
        ''')

        self.__test_query('Join 3: Order Details', '''
            SELECT
                o_orderkey,
                o_orderdate,
                l_linenumber,
                l_quantity,
                l_extendedprice
            FROM orders
            JOIN lineitem ON o_orderkey = l_orderkey
            WHERE o_orderdate BETWEEN date '1996-01-01' AND date '1996-03-31'
                AND l_quantity > 30
            ORDER BY o_orderdate, o_orderkey
            LIMIT 500
        ''')

        self.__test_query('Join 4: Supplier Orders', '''
            SELECT
                s_name,
                s_address,
                COUNT(DISTINCT l_orderkey) as order_count,
                SUM(l_extendedprice * (1 - l_discount)) as revenue
            FROM supplier
            JOIN lineitem ON s_suppkey = l_suppkey
            WHERE l_shipdate >= date '1996-01-01'
                AND l_shipdate < date '1997-01-01'
            GROUP BY s_suppkey, s_name, s_address
            ORDER BY revenue DESC
            LIMIT 50
        ''')

        self.__test_query('Join 5: Customer Nation Analysis', '''
            SELECT
                r.r_name AS region_name,
                n.n_name AS nation_name,
                COUNT(DISTINCT c.c_custkey) AS customer_count,
                COUNT(o.o_orderkey) AS total_orders,
                AVG(o.o_totalprice) AS avg_order_value
            FROM region AS r
            JOIN nation AS n ON n.n_regionkey = r.r_regionkey
            JOIN customer AS c ON c.c_nationkey = n.n_nationkey
            JOIN orders AS o ON o.o_custkey = c.c_custkey
            WHERE o.o_orderdate >= date '1997-01-01'
            GROUP BY r.r_name, n.n_name
            ORDER BY total_orders DESC
        ''')

        self.__test_query('Join 6: Part Lineitem Summary', '''
            SELECT
                p_brand,
                p_container,
                COUNT(*) as shipment_count,
                AVG(l_quantity) as avg_quantity,
                SUM(l_extendedprice) as total_value
            FROM part
            JOIN lineitem ON p_partkey = l_partkey
            WHERE l_shipdate >= date '1995-01-01'
                AND p_size < 15
            GROUP BY p_brand, p_container
            HAVING COUNT(*) > 10
        ''')

        # ========================================================================
        # CATEGORY 3: Complex Multi-table Joins
        # ========================================================================

        self.__test_query('Complex Join 1: Customer Segment Revenue', '''
            SELECT
                c_mktsegment,
                AVG(l_extendedprice * (1 - l_discount)) as avg_revenue,
                SUM(l_extendedprice * (1 - l_discount)) as total_revenue,
                COUNT(DISTINCT c_custkey) as customer_count
            FROM customer
            JOIN orders ON c_custkey = o_custkey
            JOIN lineitem ON o_orderkey = l_orderkey
            WHERE l_shipdate >= date '1995-06-01'
                AND l_shipdate < date '1995-09-01'
                AND c_mktsegment = 'BUILDING'
            GROUP BY c_mktsegment
        ''')

        self.__test_query('Complex Join 2: Supplier Revenue Analysis', '''
            SELECT
                r.r_name AS region_name,
                sn.n_name AS supplier_nation,
                SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM region AS r
            JOIN nation AS cn ON cn.n_regionkey = r.r_regionkey
            JOIN customer AS c ON c.c_nationkey = cn.n_nationkey
            JOIN orders AS o ON o.o_custkey = c.c_custkey
            JOIN lineitem AS l ON l.l_orderkey = o.o_orderkey
            JOIN supplier AS s ON s.s_suppkey = l.l_suppkey
            JOIN nation AS sn ON sn.n_nationkey = s.s_nationkey
            WHERE sn.n_regionkey = r.r_regionkey
                AND o.o_orderdate >= date '1994-01-01'
                AND o.o_orderdate < date '1995-01-01'
            GROUP BY r.r_name, sn.n_name
            ORDER BY revenue DESC
            LIMIT 100
        ''')

        self.__test_query('Complex Join 3: Part Supplier Customer Chain', '''
            SELECT
                p_brand,
                c_mktsegment,
                COUNT(DISTINCT o_orderkey) as order_count,
                SUM(l_quantity) as total_quantity,
                AVG(l_discount) as avg_discount
            FROM part
            JOIN lineitem ON p_partkey = l_partkey
            JOIN orders ON l_orderkey = o_orderkey
            JOIN customer ON o_custkey = c_custkey
            WHERE p_type LIKE '%BRASS%'
                AND o_orderdate >= date '1996-01-01'
                AND o_orderdate < date '1997-01-01'
            GROUP BY p_brand, c_mktsegment
            ORDER BY order_count DESC
        ''')

        self.__test_query('Complex Join 4: Multi-way with Partsupp', '''
            SELECT
                p_partkey,
                s_name,
                ps_supplycost,
                SUM(l_quantity) as total_shipped
            FROM part
            JOIN partsupp ON p_partkey = ps_partkey
            JOIN supplier ON ps_suppkey = s_suppkey
            JOIN lineitem ON p_partkey = l_partkey AND s_suppkey = l_suppkey
            WHERE l_shipdate >= date '1996-01-01'
                AND l_shipdate < date '1996-06-01'
                AND p_size > 15
            GROUP BY p_partkey, s_suppkey, s_name, ps_supplycost
            HAVING SUM(l_quantity) > 50
            LIMIT 100
        ''')

        self.__test_query('Complex Join 5: Full Chain Analysis', '''
            SELECT
                c_mktsegment,
                p_brand,
                COUNT(*) as transaction_count,
                AVG(l_extendedprice * (1 - l_discount)) as avg_net_price
            FROM customer
            JOIN orders ON c_custkey = o_custkey
            JOIN lineitem ON o_orderkey = l_orderkey
            JOIN part ON l_partkey = p_partkey
            WHERE o_orderdate >= date '1997-01-01'
                AND o_orderdate < date '1997-07-01'
                AND l_discount > 0.05
            GROUP BY c_mktsegment, p_brand
            HAVING COUNT(*) > 5
        ''')

        self.__test_query('Complex Join 6: Regional Supply Chain', '''
            SELECT
                sn.n_name AS supplier_nation,
                cn.n_name AS customer_nation,
                r.r_name AS region_name,
                COUNT(DISTINCT o.o_orderkey) AS orders,
                SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            FROM region AS r
            JOIN nation AS cn ON cn.n_regionkey = r.r_regionkey
            JOIN customer AS c ON c.c_nationkey = cn.n_nationkey
            JOIN orders AS o ON o.o_custkey = c.c_custkey
            JOIN lineitem AS l ON l.l_orderkey = o.o_orderkey
            JOIN supplier AS s ON s.s_suppkey = l.l_suppkey
            JOIN nation AS sn ON sn.n_nationkey = s.s_nationkey
            WHERE sn.n_regionkey = r.r_regionkey
                AND o.o_orderdate >= date '1995-01-01'
                AND o.o_orderdate < date '1996-01-01'
            GROUP BY sn.n_name, cn.n_name, r.r_name
            HAVING COUNT(DISTINCT o.o_orderkey) > 10
            ORDER BY revenue DESC
            LIMIT 50
        ''')

        # ========================================================================
        # CATEGORY 4: Selective Scans with Filters
        # ========================================================================

        self.__test_query('Selective 1: Discount Range', '''
            SELECT
                l_orderkey,
                l_linenumber,
                l_quantity,
                l_extendedprice,
                l_discount,
                l_extendedprice * l_discount as discount_revenue
            FROM lineitem
            WHERE l_discount BETWEEN 0.05 AND 0.07
                AND l_quantity < 24
                AND l_shipdate >= date '1994-01-01'
            ORDER BY discount_revenue DESC
            LIMIT 100
        ''')

        self.__test_query('Selective 2: High Value Orders', '''
            SELECT
                o_orderkey,
                o_custkey,
                o_totalprice,
                o_orderdate
            FROM orders
            WHERE o_totalprice > 300000
                AND o_orderdate >= date '1995-01-01'
                AND o_orderdate < date '1997-01-01'
            ORDER BY o_totalprice DESC
            LIMIT 50
        ''')

        self.__test_query('Selective 3: Premium Customers', '''
            SELECT
                c_custkey,
                c_name,
                c_acctbal,
                c_mktsegment
            FROM customer
            WHERE c_acctbal > 8000
                AND c_mktsegment IN ('AUTOMOBILE', 'MACHINERY')
            ORDER BY c_acctbal DESC
            LIMIT 100
        ''')

        self.__test_query('Selective 4: Specific Part Types', '''
            SELECT
                p_partkey,
                p_name,
                p_brand,
                p_retailprice
            FROM part
            WHERE p_brand = 'Brand#23'
                AND p_container IN ('SM BOX', 'SM PACK')
                AND p_size BETWEEN 5 AND 15
            ORDER BY p_retailprice DESC
        ''')

        self.__test_query('Selective 5: Late Shipments', '''
            SELECT
                l_orderkey,
                l_linenumber,
                l_shipdate,
                l_commitdate,
                l_receiptdate
            FROM lineitem
            WHERE l_shipdate > l_commitdate
                AND l_receiptdate >= date '1996-01-01'
                AND l_receiptdate < date '1996-06-01'
            ORDER BY l_shipdate
            LIMIT 200
        ''')

        self.__test_query('Selective 6: Low Supply Cost', '''
            SELECT
                ps_partkey,
                ps_suppkey,
                ps_supplycost,
                ps_availqty
            FROM partsupp
            WHERE ps_supplycost < 50
                AND ps_availqty > 5000
            ORDER BY ps_supplycost, ps_availqty DESC
            LIMIT 150
        ''')

        # ========================================================================
        # CATEGORY 5: Subquery Patterns
        # ========================================================================

        self.__test_query('Subquery 1: High Value Customers', '''
            SELECT
                c_custkey,
                c_name,
                c_acctbal,
                c_mktsegment
            FROM customer
            WHERE c_custkey IN (
                SELECT o_custkey
                FROM orders
                WHERE o_totalprice > 200000
                    AND o_orderdate >= date '1995-01-01'
            )
            ORDER BY c_acctbal DESC
            LIMIT 50
        ''')

        self.__test_query('Subquery 2: Frequent Buyers', '''
            SELECT
                c_custkey,
                c_name,
                c_phone
            FROM customer
            WHERE c_custkey IN (
                SELECT o_custkey
                FROM orders
                WHERE o_orderdate >= date '1997-01-01'
                GROUP BY o_custkey
                HAVING COUNT(*) > 10
            )
            ORDER BY c_name
            LIMIT 100
        ''')

        self.__test_query('Subquery 3: Popular Parts', '''
            SELECT
                p_partkey,
                p_name,
                p_brand,
                p_retailprice
            FROM part
            WHERE p_partkey IN (
                SELECT l_partkey
                FROM lineitem
                WHERE l_shipdate >= date '1997-01-01'
                    AND l_shipdate < date '1998-01-01'
                GROUP BY l_partkey
                HAVING SUM(l_quantity) > 500
            )
            ORDER BY p_retailprice DESC
        ''')

        self.__test_query('Subquery 4: Top Suppliers by Volume', '''
            SELECT
                s_suppkey,
                s_name,
                s_address,
                s_phone
            FROM supplier
            WHERE s_suppkey IN (
                SELECT l_suppkey
                FROM lineitem
                WHERE l_shipdate >= date '1996-01-01'
                GROUP BY l_suppkey
                HAVING SUM(l_quantity) > 10000
            )
            ORDER BY s_name
        ''')

        self.__test_query('Subquery 5: Orders Above Average', '''
            SELECT
                o_orderkey,
                o_custkey,
                o_totalprice,
                o_orderdate
            FROM orders
            WHERE o_totalprice > (
                SELECT AVG(o_totalprice)
                FROM orders
                WHERE o_orderdate >= date '1996-01-01'
            )
            AND o_orderdate >= date '1997-01-01'
            ORDER BY o_totalprice DESC
            LIMIT 100
        ''')

        self.__test_query('Subquery 6: Customers with Recent Large Orders', '''
            SELECT
                c_custkey,
                c_name,
                c_mktsegment
            FROM customer
            WHERE EXISTS (
                SELECT 1
                FROM orders
                WHERE o_custkey = c_custkey
                    AND o_totalprice > 250000
                    AND o_orderdate >= date '1997-01-01'
            )
            ORDER BY c_name
            LIMIT 50
        ''')

        # ========================================================================
        # CATEGORY 6: Large Scans with Sorting
        # ========================================================================

        self.__test_query('Large Scan 1: Sorted Lineitem by Price', '''
            SELECT
                l_orderkey,
                l_partkey,
                l_suppkey,
                l_quantity,
                l_extendedprice
            FROM lineitem
            WHERE l_shipdate >= date '1997-01-01'
                AND l_shipdate < date '1997-04-01'
            ORDER BY l_extendedprice DESC
            LIMIT 200
        ''')

        self.__test_query('Large Scan 2: Orders by Date', '''
            SELECT
                o_orderkey,
                o_custkey,
                o_totalprice,
                o_orderdate,
                o_orderpriority
            FROM orders
            WHERE o_orderdate >= date '1996-01-01'
                AND o_orderdate < date '1997-01-01'
            ORDER BY o_orderdate DESC, o_totalprice DESC
            LIMIT 500
        ''')

        self.__test_query('Large Scan 3: Parts by Price', '''
            SELECT
                p_partkey,
                p_name,
                p_brand,
                p_type,
                p_retailprice
            FROM part
            WHERE p_retailprice > 1000
            ORDER BY p_retailprice DESC, p_partkey
            LIMIT 300
        ''')

        self.__test_query('Large Scan 4: Customer Balance Ranking', '''
            SELECT
                c.c_custkey,
                c.c_name,
                c.c_acctbal,
                c.c_mktsegment,
                n.n_name AS nation_name,
                r.r_name AS region_name
            FROM customer AS c
            JOIN nation AS n ON c.c_nationkey = n.n_nationkey
            JOIN region AS r ON n.n_regionkey = r.r_regionkey
            WHERE c.c_acctbal > 0
            ORDER BY c.c_acctbal DESC, c.c_name
            LIMIT 400
        ''')

        self.__test_query('Large Scan 5: Lineitem Quantity Sort', '''
            SELECT
                l_orderkey,
                l_partkey,
                l_quantity,
                l_extendedprice,
                l_discount
            FROM lineitem
            WHERE l_shipdate >= date '1996-06-01'
                AND l_shipdate < date '1996-09-01'
                AND l_quantity > 40
            ORDER BY l_quantity DESC, l_extendedprice DESC
            LIMIT 250
        ''')

        self.__test_query('Large Scan 6: Recent Shipments', '''
            SELECT
                l_orderkey,
                l_linenumber,
                l_shipdate,
                l_receiptdate,
                l_extendedprice * (1 - l_discount) as net_price
            FROM lineitem
            WHERE l_shipdate >= date '1998-06-01'
            ORDER BY l_shipdate DESC, net_price DESC
            LIMIT 300
        ''')

        # ========================================================================
        # CATEGORY 7: Aggregation with HAVING
        # ========================================================================

        self.__test_query('Having 1: Large Order Aggregates', '''
            SELECT
                l_orderkey,
                SUM(l_quantity) as total_qty,
                SUM(l_extendedprice) as total_price,
                COUNT(*) as line_count
            FROM lineitem
            GROUP BY l_orderkey
            HAVING SUM(l_quantity) > 300
            ORDER BY total_price DESC
            LIMIT 100
        ''')

        self.__test_query('Having 2: High Volume Customers', '''
            SELECT
                o_custkey,
                COUNT(*) as order_count,
                SUM(o_totalprice) as total_spent,
                AVG(o_totalprice) as avg_order_value
            FROM orders
            WHERE o_orderdate >= date '1996-01-01'
            GROUP BY o_custkey
            HAVING COUNT(*) > 15
            ORDER BY total_spent DESC
            LIMIT 50
        ''')

        self.__test_query('Having 3: Popular Parts by Brand', '''
            SELECT
                p_brand,
                COUNT(*) as part_count,
                AVG(p_retailprice) as avg_price,
                MIN(p_retailprice) as min_price,
                MAX(p_retailprice) as max_price
            FROM part
            GROUP BY p_brand
            HAVING COUNT(*) > 50
            ORDER BY avg_price DESC
        ''')

        self.__test_query('Having 4: High Revenue Suppliers', '''
            SELECT
                l_suppkey,
                COUNT(DISTINCT l_orderkey) as order_count,
                SUM(l_extendedprice * (1 - l_discount)) as total_revenue,
                AVG(l_quantity) as avg_quantity
            FROM lineitem
            WHERE l_shipdate >= date '1997-01-01'
            GROUP BY l_suppkey
            HAVING SUM(l_extendedprice * (1 - l_discount)) > 500000
            ORDER BY total_revenue DESC
            LIMIT 75
        ''')

        self.__test_query('Having 5: Part Categories with High Sales', '''
            SELECT
                p_type,
                COUNT(DISTINCT l_orderkey) as order_count,
                SUM(l_quantity) as total_quantity,
                AVG(l_extendedprice) as avg_price
            FROM part
            JOIN lineitem ON p_partkey = l_partkey
            WHERE l_shipdate >= date '1996-01-01'
                AND l_shipdate < date '1997-01-01'
            GROUP BY p_type
            HAVING SUM(l_quantity) > 1000
            ORDER BY total_quantity DESC
            LIMIT 25
        ''')

        self.__test_query('Having 6: Customer Segments with Volume', '''
            SELECT
                c_mktsegment,
                COUNT(DISTINCT c_custkey) as customer_count,
                COUNT(o_orderkey) as total_orders,
                AVG(o_totalprice) as avg_order_price
            FROM customer
            JOIN orders ON c_custkey = o_custkey
            WHERE o_orderdate >= date '1997-01-01'
            GROUP BY c_mktsegment
            HAVING COUNT(o_orderkey) > 1000
            ORDER BY total_orders DESC
        ''')
