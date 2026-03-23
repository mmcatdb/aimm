from common.query_registry import query
from common.drivers import DriverType
from datasets.tpch.tpch_database import TpchDatabase

class TpchPostgresDatabase(TpchDatabase[str]):

    def __init__(self):
        super().__init__(DriverType.POSTGRES)

    @query('basic', 1.0, 'TPC-H Query 1 variations')
    def _pricing_report(self):
        return f'''
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
            WHERE l_shipdate <= date '1998-12-01' - interval '{self._param_int('delta', 60, 120)} days'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        '''

    @query('basic', 1.0, 'TPC-H Query 3 variations')
    def _orders(self):
        date = self._param_date()

        return f'''
            SELECT
                l_orderkey,
                SUM(l_extendedprice * (1 - l_discount)) as revenue,
                o_orderdate,
                o_shippriority
            FROM customer, orders, lineitem
            WHERE c_mktsegment = '{self._param_segment()}'
                AND c_custkey = o_custkey
                AND l_orderkey = o_orderkey
                AND o_orderdate < date '{date}'
                AND l_shipdate > date '{date}'
            GROUP BY l_orderkey, o_orderdate, o_shippriority
            ORDER BY revenue DESC, o_orderdate
            LIMIT 10
        '''

    @query('basic', 1.0, 'TPC-H Query 5 variations')
    def _supplier_volume(self):
        date = self._param_date()

        return f'''
            SELECT
                SUM(l_extendedprice * (1 - l_discount)) as revenue
            FROM customer, orders, lineitem, supplier
            WHERE c_custkey = o_custkey
                AND l_orderkey = o_orderkey
                AND l_suppkey = s_suppkey
                AND c_nationkey = s_nationkey
                AND o_orderdate >= date '{date}'
                AND o_orderdate < date '{date}' + interval '1 year'
            GROUP BY c_nationkey
            ORDER BY revenue DESC
        '''

    @query('basic', 1.0, 'TPC-H Query 6 variations')
    def _forecast_revenue(self):
        date = self._param_date()
        discount = self._param_float('discount', 0.02, 0.09)

        return f'''
            SELECT
                SUM(l_extendedprice * l_discount) as revenue
            FROM lineitem
            WHERE l_shipdate >= date '{date}'
                AND l_shipdate < date '{date}' + interval '1 year'
                AND l_discount BETWEEN {discount} - 0.01 AND {discount} + 0.01
                AND l_quantity < {self._param_int('quantity', 20, 30)}
        '''

    @query('basic', 1.0, 'TPC-H Query 10 variations')
    def _customer_revenue(self):
        date = self._param_date()

        return f'''
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
                AND o_orderdate >= date '{date}'
                AND o_orderdate < date '{date}' + interval '3 months'
                AND l_returnflag = 'R'
            GROUP BY c_custkey, c_name, c_acctbal, c_phone, c_address, c_comment
            ORDER BY revenue DESC
            LIMIT 20
        '''

    @query('basic', 1.0, 'TPC-H Query 12 variations')
    def _ship_mode_analysis(self):
        date = self._param_date()

        return f'''
            SELECT
                l_shipmode,
                SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) as high_line_count,
                SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) as low_line_count
            FROM orders, lineitem
            WHERE o_orderkey = l_orderkey
                AND l_shipmode IN ({self._param_shipmodes()})
                AND l_commitdate < l_receiptdate
                AND l_shipdate < l_commitdate
                AND l_receiptdate >= date '{date}'
                AND l_receiptdate < date '{date}' + interval '1 year'
            GROUP BY l_shipmode
            ORDER BY l_shipmode
        '''

    #region Simple Aggregation

    @query('agg', 1.0, 'Lineitem Summary')
    def lineitem_summary(self):
        return '''
            SELECT
                l_returnflag,
                COUNT(*) as count,
                AVG(l_quantity) as avg_qty,
                SUM(l_extendedprice) as total_price
            FROM lineitem
            WHERE l_shipdate <= date '1998-08-01'
            GROUP BY l_returnflag
        '''

    @query('agg', 1.0, 'Order Stats')
    def order_statistics(self):
        return '''
            SELECT
                o_orderpriority,
                COUNT(*) as order_count,
                AVG(o_totalprice) as avg_price,
                MIN(o_totalprice) as min_price,
                MAX(o_totalprice) as max_price
            FROM orders
            WHERE o_orderdate >= date '1996-01-01'
            GROUP BY o_orderpriority
        '''

    @query('agg', 1.0, 'Customer Segments')
    def customer_segments(self):
        return '''
            SELECT
                c_mktsegment,
                COUNT(*) as customer_count,
                AVG(c_acctbal) as avg_balance,
                SUM(c_acctbal) as total_balance
            FROM customer
            WHERE c_acctbal > 0
            GROUP BY c_mktsegment
            ORDER BY customer_count DESC
        '''

    @query('agg', 1.0, 'Part Analysis')
    def part_analysis(self):
        return '''
            SELECT
                p_brand,
                p_type,
                COUNT(*) as part_count,
                AVG(p_retailprice) as avg_price
            FROM part
            WHERE p_size BETWEEN 10 AND 30
            GROUP BY p_brand, p_type
            HAVING COUNT(*) > 5
        '''

    @query('agg', 1.0, 'Supplier Stats')
    def supplier_stats(self):
        return '''
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
        '''

    @query('agg', 1.0, 'Discount Analysis')
    def discount_analysis(self):
        return '''
            SELECT
                l_linestatus,
                AVG(l_discount) as avg_discount,
                AVG(l_tax) as avg_tax,
                COUNT(*) as line_count
            FROM lineitem
            WHERE l_shipdate >= date '1997-01-01'
                AND l_shipdate < date '1998-01-01'
            GROUP BY l_linestatus
        '''

    #endregion
    #region Simple Join

    @query('join', 1.0, 'Customer Orders')
    def customer_orders(self):
        return '''
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
        '''

    @query('join', 1.0, 'Parts and Suppliers')
    def parts_and_suppliers(self):
        return '''
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
        '''

    @query('join', 1.0, 'Order Details')
    def order_details(self):
        return '''
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
        '''

    @query('join', 1.0, 'Supplier Orders')
    def supplier_orders(self):
        return '''
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
        '''

    @query('join', 1.0, 'Customer Nation Analysis')
    def customer_nation_analysis(self):
        return '''
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
        '''

    @query('join', 1.0, 'Part Lineitem Summary')
    def part_lineitem_summary(self):
        return '''
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
        '''

    #endregion
    #region Multi-table Join

    @query('complex-join', 1.0, 'Customer Segment Revenue')
    def customer_segment_revenue(self):
        return '''
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
        '''

    @query('complex-join', 1.0, 'Supplier Revenue Analysis')
    def supplier_revenue_analysis(self):
        return '''
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
        '''

    @query('complex-join', 1.0, 'Part Supplier Customer Chain')
    def part_supplier_customer_chain(self):
        return '''
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
        '''

    @query('complex-join', 1.0, 'Multi-way with Partsupp')
    def multi_way_with_partsupp(self):
        return '''
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
        '''

    @query('complex-join', 1.0, 'Full Chain Analysis')
    def full_chain_analysis(self):
        return '''
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
        '''

    @query('complex-join', 1.0, 'Regional Supply Chain')
    def regional_supply_chain(self):
        return '''
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
        '''

    #endregion
    #region Selective Scan + Filter

    @query('selective', 1.0, 'Discount Range')
    def discount_range(self):
        return '''
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
        '''

    @query('selective', 1.0, 'High Value Orders')
    def high_value_orders(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Premium Customers')
    def premium_customers(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Specific Part Types')
    def specific_part_types(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Late Shipments')
    def late_shipments(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Low Supply Cost')
    def low_supply_cost(self):
        return '''
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
        '''

    #endregion
    #region Subquery Pattern

    @query('subquery', 1.0, 'High Value Customers')
    def high_value_customers(self):
        return '''
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
        '''

    @query('subquery', 1.0, 'Frequent Buyers')
    def frequent_buyers(self):
        return '''
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
        '''

    @query('subquery', 1.0, 'Popular Parts')
    def popular_parts(self):
        return '''
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
        '''

    @query('subquery', 1.0, 'Top Suppliers by Volume')
    def top_suppliers_by_volume(self):
        return '''
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
        '''

    @query('subquery', 1.0, 'Orders Above Average')
    def orders_above_average(self):
        return '''
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
        '''

    @query('subquery', 1.0, 'Customers with Recent Large Orders')
    def customers_large_orders(self):
        return '''
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
        '''

    #endregion
    #region Large Scan + Sorting

    @query('large-scan', 1.0, 'Lineitems by Price')
    def lineitem_by_price(self):
        return '''
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
        '''

    @query('large-scan', 1.0, 'Orders by Date')
    def orders_by_date(self):
        return '''
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
        '''

    @query('large-scan', 1.0, 'Parts by Price')
    def parts_by_price(self):
        return '''
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
        '''

    @query('large-scan', 1.0, 'Customer Balance Ranking')
    def customer_balance_ranking(self):
        return '''
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
        '''

    @query('large-scan', 1.0, 'Lineitem Quantity Sort')
    def lineitem_quantity_sort(self):
        return '''
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
        '''

    @query('large-scan', 1.0, 'Recent Shipments')
    def recent_shipments(self):
        return '''
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
        '''

    #endregion
    #region Aggregation + HAVING

    @query('having', 1.0, 'Large Order Aggregates')
    def large_order_aggregates(self):
        return '''
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
        '''

    @query('having', 1.0, 'High Volume Customers')
    def high_volume_customers(self):
        return '''
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
        '''

    @query('having', 1.0, 'Popular Parts by Brand')
    def popular_parts_by_brand(self):
        return '''
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
        '''

    @query('having', 1.0, 'High Revenue Suppliers')
    def high_revenue_suppliers(self):
        return '''
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
        '''

    @query('having', 1.0, 'Part Categories with High Sales')
    def part_categories_high_sales(self):
        return '''
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
        '''

    @query('having', 1.0, 'Customer Segments with Volume')
    def customer_segments_volume(self):
        return '''
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
        '''

    #endregion
