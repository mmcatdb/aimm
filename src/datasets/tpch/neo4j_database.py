from common.query_registry import query
from common.drivers import DriverType
from datasets.tpch.tpch_database import TpchDatabase

class TpchNeo4jDatabase(TpchDatabase[str]):

    def __init__(self):
        super().__init__(DriverType.NEO4J)

    #region Join

    @query('join', 1.0, 'Pricing Summary Report')
    def _pricing_report(self):
        return f'''
            MATCH (li:LineItem)
            WHERE li.l_shipdate <= date('{self._param_date(1996)}')
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
        '''

    @query('join', 1.0, 'Local Supplier Volume')
    def _supplier_volume(self):
        date = self._param_date()

        return f'''
            MATCH (r:Region {{r_name: '{self._param_region()}'}})<-[:IN_REGION]-(n:Nation)
            MATCH (n)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)
            MATCH (n)<-[:IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(li)
            WHERE o.o_orderdate >= date('{date}')
                AND o.o_orderdate < date('{date}') + duration('P1Y')
            WITH n.n_name AS nation_name,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            RETURN nation_name, revenue
            ORDER BY revenue DESC
        '''

    @query('join', 1.0, 'Forecasting Revenue Change')
    def _forecast_revenue(self):
        date = self._param_date()
        discount = self._param_float('discount', 0.02, 0.09)

        return f'''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('{date}')
                AND li.l_shipdate < date('{date}') + duration('P1Y')
                AND li.l_discount >= {discount} - 0.01
                AND li.l_discount <= {discount} + 0.01
                AND li.l_quantity < {self._param_int('quantity', 20, 30)}
            RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
        '''

    @query('join', 1.0, 'Returned Item Reporting (Fix: Date generation)')
    def _returned_items(self):
        date = self._param_date(1993, 1995)

        return f'''
            MATCH (n:Nation)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)
            WHERE o.o_orderdate >= date('{date}')
                AND o.o_orderdate < date('{date}') + duration('P3M')
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
        '''

    @query('join', 1.0, 'Shipping Modes and Order Priority (Fix: Shipmode generation)')
    def _shipping_priority(self):
        date = self._param_date()

        return f'''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)
            WHERE li.l_shipmode IN [{self._param_shipmodes()}]
                AND li.l_commitdate < li.l_receiptdate
                AND li.l_shipdate < li.l_commitdate
                AND li.l_receiptdate >= date('{date}')
                AND li.l_receiptdate < date('{date}') + duration('P1Y')
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
        '''

    @query('join', 1.0, 'Promotion Effect (Fix: Month range and end date logic)')
    def _promotion_effect(self):
        date = self._param_date()

        return f'''
            MATCH (li:LineItem)-[:OF_PART]->(p:Part)
            WHERE li.l_shipdate >= date('{date}')
                AND li.l_shipdate < date('{date}') + duration('P1M')
            WITH sum(
            CASE
                WHEN p.p_type STARTS WITH 'PROMO'
                THEN li.l_extendedprice * (1 - li.l_discount)
                ELSE 0
            END
            ) AS promo_revenue,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue
            RETURN 100.00 * promo_revenue / total_revenue AS promo_revenue_percentage
        '''

    @query('join', 1.0, 'Discounted Revenue (Fix: Brand generation)')
    def _discounted_revenue(self):
        qty1 = self._param_int('qty1', 1, 10)
        qty2 = self._param_int('qty2', 10, 20)
        qty3 = self._param_int('qty3', 20, 30)

        return f'''
            MATCH (li:LineItem)-[:OF_PART]->(p:Part)
            WHERE li.l_shipinstruct = 'DELIVER IN PERSON'
                AND li.l_shipmode IN ['AIR', 'AIR REG']
                AND ((
                    p.p_brand = '{self._param_brand('brand1')}'
                    AND p.p_container IN ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']
                    AND li.l_quantity >= {qty1} AND li.l_quantity <= {qty1} + 10
                    AND p.p_size >= 1 AND p.p_size <= 5
                ) OR (
                    p.p_brand = '{self._param_brand('brand2')}'
                    AND p.p_container IN ['MED BAG', 'MED BOX', 'MED PKG', 'MED PACK']
                    AND li.l_quantity >= {qty2} AND li.l_quantity <= {qty2} + 10
                    AND p.p_size >= 1 AND p.p_size <= 10
                ) OR (
                    p.p_brand = '{self._param_brand('brand3')}'
                    AND p.p_container IN ['LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
                    AND li.l_quantity >= {qty3} AND li.l_quantity <= {qty3} + 10
                    AND p.p_size >= 1 AND p.p_size <= 15
                ))
            RETURN sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
        '''

    @query('join', 1.0, 'Customer Orders')
    def _customer_orders(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
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
        '''

    @query('join', 1.0, 'Parts and Suppliers')
    def _parts_and_suppliers(self):
        return '''
            MATCH (ps:PartSupp)-[:FOR_PART]->(p:Part)
            WHERE p.p_size > 20 AND ps.ps_supplycost < 100
            RETURN
                p.p_partkey AS partkey,
                p.p_name AS name,
                ps.ps_supplycost AS supplycost,
                ps.ps_availqty AS availqty
            ORDER BY supplycost
            LIMIT 200
        '''

    @query('join', 1.0, 'Order Details')
    def _order_details(self):
        return '''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)
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
        '''

    @query('join', 1.0, 'Supplier Orders')
    def _supplier_orders(self):
        return '''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
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
        '''

    @query('join', 1.0, 'Customer Nation Analysis')
    def _customer_nation_analysis(self):
        return '''
            MATCH (r:Region)<-[:IN_REGION]-(n:Nation)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)
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
        '''

    @query('join', 1.0, 'Part Lineitem Summary')
    def _part_lineitem_summary(self):
        return '''
            MATCH (p:Part)<-[:OF_PART]-(li:LineItem)
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
        '''

    #endregion
    #region Scan

    @query('scan', 1.0, 'Simple Scan (All Nodes of a Label)')
    def _simple_scan(self):
        # TODO param
        return f'MATCH (n:Nation) RETURN n.n_name, n.n_comment LIMIT 100'

    @query('scan', 1.0, 'Scan with Exact Property Match')
    def _exact_match(self):
        return f'''
            MATCH (c:Customer {{c_name: '{self._param_customer()}'}})
            RETURN c.c_address, c.c_phone
        '''

    @query('scan', 1.0, 'Scan with Numeric Filter')
    def _numeric_filter(self):
        return f'MATCH (s:Supplier) WHERE s.s_acctbal > {self._param_int("balance", 5000, 9500)} RETURN s.s_name, s.s_acctbal'

    @query('scan', 1.0, 'Scan with Sort')
    def _sort(self):
        # TODO param
        return f'MATCH (p:Part) RETURN p.p_name, p.p_retailprice ORDER BY p.p_retailprice DESC LIMIT 50'

    @query('scan', 1.0, 'Scan with Sort + Limit')
    def _sort_limit(self):
        return f'MATCH (o:Orders) RETURN o.o_orderkey, o.o_totalprice ORDER BY o.o_totalprice DESC LIMIT {self._param_int("limit", 10, 50)}'

    @query('scan', 1.0, 'Scan with IN list')
    def _in_list(self):
        return f'MATCH (p:Part) WHERE p.p_container IN [{self._param_containers()}] RETURN p.p_name, p.p_container'

    @query('scan', 1.0, 'Scan with STARTS WITH')
    def _starts_with(self):
        return f'''
            MATCH (p:Part)
            WHERE p.p_type STARTS WITH '{self._param_part_type()}'
            RETURN p.p_name, p.p_type
        '''

    @query('scan', 1.0, 'Scan with Date Filter')
    def _date_filter(self):
        return f'''
            MATCH (o:Orders)
            WHERE o.o_orderdate > date('{self._param_date()}')
            RETURN o.o_orderkey, o.o_orderdate
        '''

    @query('scan', 1.0, 'Scan with AND')
    def _and_filter(self):
        return f'''
            MATCH (p:Part)
            WHERE p.p_size > {self._param_int('size', 10, 40)}
            AND p.p_retailprice < {self._param_int('price', 1000, 1500)}
            RETURN p.p_name, p.p_size, p.p_retailprice
        '''

    @query('scan', 1.0, 'Scan with OR')
    def _or_filter(self):
        region1 = self._param_region('region1')
        region2 = self._param_region('region2', exclude=region1)

        return f'''
            MATCH (r:Region)
            WHERE r.r_name = '{region1}'
                OR r.r_name = '{region2}'
            RETURN r.r_name
        '''

    @query('scan', 1.0, 'Scan with NOT')
    def _not_filter(self):
        return f'''
            MATCH (o:Orders)
            WHERE NOT o.o_orderstatus = '{self._param_order_status()}'
            RETURN o.o_orderkey, o.o_orderstatus
            LIMIT 100
        '''

    #endregion
    #region Aggregation

    @query('agg', 1.0, 'Count Aggregation (All)')
    def _count_agg(self):
        # TODO param
        return f'MATCH (c:Customer) RETURN count(c) AS total_customers'

    @query('agg', 1.0, 'Group-by Aggregation (Simple)')
    def _group_by_agg(self):
        # TODO param
        return f'MATCH (o:Orders) RETURN o.o_orderstatus, count(o) AS order_count ORDER BY order_count DESC'

    @query('agg', 1.0, 'Simple AVG/SUM Aggregation')
    def _avg_sum_agg(self):
        # TODO param
        return f'MATCH (li:LineItem) RETURN sum(li.l_quantity) AS total_qty, avg(li.l_extendedprice) AS avg_price, min(li.l_discount) AS min_discount'

    @query('agg', 1.0, 'Simple DISTINCT')
    def _distinct_agg(self):
        # TODO param
        return f'MATCH (c:Customer) RETURN count(DISTINCT c.c_mktsegment) AS market_segments'

    @query('agg', 1.0, 'Lineitem Summary')
    def _lineitem_summary(self):
        return '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate <= date('1998-08-01')
            RETURN
                li.l_returnflag AS returnflag,
                count(li) AS count,
                avg(li.l_quantity) AS avg_qty,
                sum(li.l_extendedprice) AS total_price
            ORDER BY returnflag
        '''

    @query('agg', 1.0, 'Order Stats')
    def _order_statistics(self):
        return '''
            MATCH (o:Orders)
            WHERE o.o_orderdate >= date('1996-01-01')
            RETURN
                o.o_orderpriority AS orderpriority,
                count(o) AS order_count,
                avg(o.o_totalprice) AS avg_price,
                min(o.o_totalprice) AS min_price,
                max(o.o_totalprice) AS max_price
            ORDER BY orderpriority
        '''

    @query('agg', 1.0, 'Customer Segments')
    def _customer_segments(self):
        return '''
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
        '''

    @query('agg', 1.0, 'Part Analysis')
    def _part_analysis(self):
        return '''
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
        '''

    @query('agg', 1.0, 'Supplier Stats')
    def _supplier_stats(self):
        return '''
            MATCH (s:Supplier)-[:IN_NATION]->(n:Nation)-[:IN_REGION]->(r:Region)
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
        '''

    @query('agg', 1.0, 'Discount Analysis')
    def _discount_analysis(self):
        return '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1997-01-01') AND li.l_shipdate < date('1998-01-01')
            RETURN
                li.l_linestatus AS linestatus,
                avg(li.l_discount) AS avg_discount,
                avg(li.l_tax) AS avg_tax,
                count(li) AS line_count
            ORDER BY linestatus
        '''

    #endregion
    #region N-Hop

    @query('n-hop', 1.0, '1-Hop Traversal (Find orders for a customer)')
    def _customer_orders_hop(self):
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
            WHERE c.c_name = '{self._param_customer()}'
            RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
        '''

    @query('n-hop', 1.0, '1-Hop with Filter and Aggregation (Count items in high-priority orders)')
    def _high_priority_orders(self):
        return f'''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)
            WHERE o.o_orderpriority = '{self._param_order_priority(first_n=2)}'
            RETURN o.o_orderkey, count(li) AS items
            ORDER BY items DESC
            LIMIT 50
        '''

    @query('n-hop', 1.0, '2-Hop Traversal (Find items for a customer)')
    def _customer_items(self):
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)
            WHERE c.c_name = '{self._param_customer()}'
            RETURN c.c_name, count(li) AS total_items
        '''

    @query('n-hop', 1.0, '2-Hop with Aggregation (Count orders per nation)')
    def _orders_per_nation(self):
        return f'''
            MATCH (n:Nation)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)
            WHERE n.n_name = '{self._param_nation()}'
            RETURN n.n_name, count(o) AS orders_from_nation
        '''

    @query('n-hop', 1.0, '3-Hop Traversal (Find parts from a supplier)')
    def _supplier_parts(self):
        return f'''
            MATCH (s:Supplier)<-[:SUPPLIED_BY]-(:PartSupp)-[:FOR_PART]->(p:Part)
            WHERE s.s_name = '{self._param_supplier()}'
            RETURN p.p_name, p.p_mfgr, p.p_retailprice
            LIMIT 100
        '''

    @query('n-hop', 1.0, '3-Hop with Property Filter (Customers in a region)')
    def _region_customers(self):
        return f'''
            MATCH (r:Region)<-[:IN_REGION]-(n:Nation)<-[:IN_NATION]-(c:Customer)
            WHERE r.r_name = '{self._param_region()}'
            RETURN c.c_name, n.n_name
            LIMIT 200
        '''

    @query('n-hop', 1.0, 'Complex Path (4-Hop) and Aggregation')
    def _customer_part_count(self):
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)-[:OF_PART]->(p:Part)
            WHERE c.c_custkey = {self._param_custkey()}
            RETURN p.p_name, count(p) AS part_count
            ORDER BY part_count DESC
            LIMIT 10
        '''

    @query('n-hop', 1.0, 'Multi-hop with CONTAINS (Find orders for a part type)')
    def _orders_for_part_type(self):
        return f'''
            MATCH (p:Part)<-[:OF_PART]-(:LineItem)<-[:HAS_ITEM]-(o:Orders)
            WHERE p.p_name CONTAINS '{self._param_part_name_word()}'
            RETURN o.o_orderkey, o.o_orderdate, o.o_totalprice
            LIMIT 50
        '''

    @query('n-hop', 1.0, 'Aggregation on Traversal (Supplier stock value)')
    def _supplier_stock_value(self):
        return f'''
            MATCH (s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)
            WHERE s.s_acctbal < {self._param_int('balance', 0, 1000)}
            RETURN s.s_name, sum(ps.ps_supplycost * ps.ps_availqty) AS stock_value
            ORDER BY stock_value DESC
            LIMIT 20
        '''

    @query('n-hop', 1.0, 'Complex Path with Multiple Filters')
    def _complex_path(self):
        return f'''
            MATCH (n:Nation {{n_name: '{self._param_nation()}'}})<-[:IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(ps:PartSupp)-[:FOR_PART]->(p:Part)
            WHERE p.p_retailprice > {self._param_int('price', 1500, 2000)}
                AND ps.ps_availqty > {self._param_int('qty', 5000, 8000)}
            RETURN s.s_name, p.p_name, ps.ps_supplycost, p.p_retailprice
            LIMIT 50
        '''

    #endregion
    #region Optional
    # Well it seems that some of these don't actually produce Optional operator, but they should be interesting nonetheless.

    @query('optional', 1.0, 'Suppliers with optional revenue from delivered items')
    def _optional_revenue(self):
        return f'''
            MATCH (s:Supplier)
            OPTIONAL MATCH (s)<-[:SUPPLIED_BY]-(l:LineItem)<-[:HAS_ITEM]-(o:Orders)
            WHERE l.l_shipdate < date('{self._param_date()}')
                AND o.o_orderpriority = '{self._param_order_priority()}'
            RETURN
                s.s_suppkey,
                count(DISTINCT o) AS orders_served,
                sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
            ORDER BY revenue DESC
            LIMIT 50
        '''

    @query('optional', 1.0, 'Nations with optional export activity')
    def _optional_export(self):
        return f'''
            MATCH (n:Nation)
            OPTIONAL MATCH (n)<-[:IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(l:LineItem)
            OPTIONAL MATCH (l)<-[:HAS_ITEM]-(o:Orders)
            WHERE l.l_quantity >= {self._param_int('quantity', 3, 17)}
                AND o.o_orderstatus = '{self._param_order_status()}'
            RETURN
                n.n_name,
                count(DISTINCT s) AS suppliers,
                count(DISTINCT o) AS orders_exported,
                sum(l.l_extendedprice) AS exported_value
            ORDER BY exported_value DESC
        '''

    @query('optional', 1.0, 'Customers with optional recent orders')
    def _optional_orders(self):
        # Should produce OptionalExpand(All) - for this one, there should be MATCH x OPTIONAL MATCH x -> y (instead of x <- y).
        # It also should be "simple" - if there is another path from Orders to Lineitem, it might not work.
        date = self._param_date()

        return f'''
            MATCH (c:Customer)
            OPTIONAL MATCH (c)-[:PLACED]->(o:Orders)
            WHERE o.o_orderdate >= date('{date}')
                AND o.o_orderdate < date('{date}') + duration('P1Y')
            RETURN
                c.c_custkey,
                count(DISTINCT o) AS orders
            ORDER BY orders DESC
        '''

    @query('optional', 1.0, 'Customers with optional supplier diversity')
    def _optional1(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
            OPTIONAL MATCH (o)-[:HAS_ITEM]->(l:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
            RETURN
                c.c_custkey,
                count(DISTINCT o) AS order_count,
                count(DISTINCT s) AS supplier_count
            ORDER BY supplier_count DESC
            LIMIT 50
        '''

    @query('optional', 1.0, 'Customers with optional regional purchasing patterns')
    def _optional2(self):
        return '''
            MATCH (c:Customer)
            OPTIONAL MATCH (c)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(l:LineItem)
            OPTIONAL MATCH (l)-[:SUPPLIED_BY]->(s:Supplier)-[:IN_NATION]->(n:Nation)-[:IN_REGION]->(r:Region)
            RETURN
                c.c_custkey,
                count(DISTINCT o) AS orders,
                collect(DISTINCT r.r_name) AS supplier_regions
            LIMIT 100
        '''

    @query('optional', 1.0, 'Orders with optional supplier competition')
    def _optional3(self):
        return '''
            MATCH (o:Orders)
            OPTIONAL MATCH (o)-[:HAS_ITEM]->(l:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
            RETURN
                o.o_orderkey,
                count(DISTINCT s) AS supplier_count,
                count(l) AS line_items
            ORDER BY supplier_count DESC
            LIMIT 100
        '''

    @query('optional', 1.0, 'Customers with optional unfulfilled supplier relationships')
    def _optional4(self):
        return '''
            MATCH (c:Customer)-[:IN_NATION]->(cn:Nation)-[:IN_REGION]->(cr:Region)
            MATCH (c)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(l:LineItem)
            OPTIONAL MATCH (l)-[:SUPPLIED_BY]->(s:Supplier)-[:IN_NATION]->(sn:Nation)-[:IN_REGION]->(sr:Region)
            WITH c, o, cr, sr
            WHERE sr <> cr OR sr IS NULL
            RETURN
                c.c_custkey,
                count(DISTINCT o) AS cross_region_orders
            ORDER BY cross_region_orders DESC
            LIMIT 50
        '''

    #endregion
    #region Basic

    @query('basic', 1.0, 'Q1 variants (Original)')
    def _basic1(self):
        return '''
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
        '''

    @query('basic', 1.0, 'Q5 variant (Original)')
    def _basic2(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem),
                (li)-[:SUPPLIED_BY]->(s:Supplier),
                (c)-[:IN_NATION]->(n:Nation)-[:IN_REGION]->(r:Region)
            WHERE r.r_name = 'ASIA'
                AND o.o_orderdate >= date('1994-01-01')
                AND o.o_orderdate < date('1995-01-01')
                AND s.s_nationkey = n.n_nationkey
            WITH n.n_name AS nation, li
            RETURN nation, sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
            ORDER BY revenue DESC
            LIMIT 3
        '''

    @query('basic', 1.0, 'Q6 variant (Original)')
    def _basic3(self):
        return '''
            MATCH (li:LineItem)
            WHERE li.l_shipdate >= date('1994-01-01')
                AND li.l_shipdate < date('1995-01-01')
                AND li.l_discount >= 0.05
                AND li.l_discount <= 0.07
                AND li.l_quantity < 24
            RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
        '''

    @query('basic', 1.0, 'Q10 variant (Original)')
    def _basic4(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem),
                (c)-[:IN_NATION]->(n:Nation)
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
        '''

    @query('basic', 1.0, 'Simple aggregation query (Original)')
    def _basic5(self):
        return '''
            MATCH (li:LineItem)
            WHERE li.l_quantity > 30
            RETURN count(li) AS count, avg(li.l_extendedprice) AS avg_price
        '''

    @query('basic', 1.0, 'Simple scan with limit (Original)')
    def _basic6(self):
        return '''
            MATCH (c:Customer)
            WHERE c.c_acctbal > 5000
            RETURN c.c_name, c.c_acctbal
            ORDER BY c.c_acctbal DESC
            LIMIT 10
        '''

    #endregion
    #region Multi-table Join

    @query('complex-join', 1.0, 'Customer Segment Revenue')
    def _customer_segment_revenue(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)
            WHERE li.l_shipdate >= date('1995-06-01')
                AND li.l_shipdate < date('1995-09-01')
                AND c.c_mktsegment = 'BUILDING'
            RETURN
                c.c_mktsegment AS mktsegment,
                avg(li.l_extendedprice * (1 - li.l_discount)) AS avg_revenue,
                sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
                count(DISTINCT c) AS customer_count
        '''

    @query('complex-join', 1.0, 'Supplier Revenue Analysis')
    def _supplier_revenue_analysis(self):
        return '''
            MATCH (r:Region)<-[:IN_REGION]-(cn:Nation)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem),
                (li)-[:SUPPLIED_BY]->(s:Supplier)-[:IN_NATION]->(sn:Nation)-[:IN_REGION]->(r)
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
        '''

    @query('complex-join', 1.0, 'Part Supplier Customer Chain')
    def _part_supplier_customer_chain(self):
        return '''
            MATCH (p:Part)<-[:OF_PART]-(li:LineItem)<-[:HAS_ITEM]-(o:Orders)<-[:PLACED]-(c:Customer)
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
        '''

    @query('complex-join', 1.0, 'Multi-way with Partsupp')
    def _multi_way_with_partsupp(self):
        return '''
            MATCH (p:Part)<-[:FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier),
                (p:Part)<-[:OF_PART]-(li:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
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
        '''

    @query('complex-join', 1.0, 'Full Chain Analysis')
    def _full_chain_analysis(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem)-[:OF_PART]->(p:Part)
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
        '''

    @query('complex-join', 1.0, 'Regional Supply Chain')
    def _regional_supply_chain(self):
        return '''
            MATCH (r:Region)<-[:IN_REGION]-(cn:Nation)<-[:IN_NATION]-(c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(li:LineItem),
                (li)-[:SUPPLIED_BY]->(s:Supplier)-[:IN_NATION]->(sn:Nation)-[:IN_REGION]->(r)
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
        '''

    #endregion
    #region Selective Scan + Filter

    @query('selective', 1.0, 'Discount Range')
    def _discount_range(self):
        return '''
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
        '''

    @query('selective', 1.0, 'High Value Orders')
    def _high_value_orders(self):
        return '''
            MATCH (o:Orders)
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
        '''

    @query('selective', 1.0, 'Premium Customers')
    def _premium_customers(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Specific Part Types')
    def _specific_part_types(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Late Shipments')
    def _late_shipments(self):
        return '''
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
        '''

    @query('selective', 1.0, 'Low Supply Cost')
    def _low_supply_cost(self):
        return '''
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
        '''

    #endregion
    #region Large Scan + Sorting

    @query('large', 1.0, 'Lineitems by Price')
    def _lineitem_by_price(self):
        return '''
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
        '''

    @query('large', 1.0, 'Orders by Date')
    def _orders_by_date(self):
        return '''
            MATCH (o:Orders)
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
        '''

    @query('large', 1.0, 'Parts by Price')
    def _parts_by_price(self):
        return '''
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
        '''

    @query('large', 1.0, 'Customer Balance Ranking')
    def _customer_balance_ranking(self):
        return '''
            MATCH (c:Customer)-[:IN_NATION]->(n:Nation)-[:IN_REGION]->(r:Region)
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
        '''

    @query('large', 1.0, 'Lineitem Quantity Sort')
    def _lineitem_quantity_sort(self):
        return '''
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
        '''

    @query('large', 1.0, 'Recent Shipments')
    def _recent_shipments(self):
        return '''
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
        '''

    #endregion
    #region Aggregation + HAVING

    @query('having', 1.0, 'Large Order Aggregates')
    def _large_order_aggregates(self):
        return '''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)
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
        '''

    @query('having', 1.0, 'High Volume Customers')
    def _high_volume_customers(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
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
        '''

    @query('having', 1.0, 'Popular Parts by Brand')
    def _popular_parts_by_brand(self):
        return '''
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
        '''

    @query('having', 1.0, 'High Revenue Suppliers')
    def _high_revenue_suppliers(self):
        return '''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
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
        '''

    @query('having', 1.0, 'Part Categories with High Sales')
    def _part_categories_high_sales(self):
        return '''
            MATCH (o:Orders)-[:HAS_ITEM]->(li:LineItem)-[:OF_PART]->(p:Part)
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
        '''

    @query('having', 1.0, 'Customer Segments with Volume')
    def _customer_segments_volume(self):
        return '''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
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
        '''

    #endregion
    #region Argument

    @query('argument', 1.0, 'Customers ordering parts they never reordered')
    def _never_reordered(self):
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)-[:HAS_ITEM]->(l:LineItem)-[:OF_PART]->(p:Part)
            WHERE NOT (c)-[:PLACED]->(:Orders)-[:HAS_ITEM]->(:LineItem)-[:OF_PART]->(p)
                AND c.c_acctbal > {self._param_int('balance', 0, 10000)}
            RETURN
                c.c_custkey,
                p.p_partkey,
                count(*) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 50
        '''

    @query('argument', 1.0, 'Parts ordered from suppliers outside their region')
    def _parts_outside_region(self):
        return f'''
            MATCH (p:Part)<-[:FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)
            MATCH (s)-[:IN_NATION]->(sn:Nation)-[:IN_REGION]->(r:Region)
            WHERE NOT (p)<-[:FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(:Supplier)-[:IN_NATION]->(:Nation)-[:IN_REGION]->(r)
                AND p.p_name CONTAINS '{self._param_part_name_word()}'
            RETURN
                p.p_partkey,
                r.r_name,
                count(*) AS cross_region_sales
            ORDER BY cross_region_sales DESC
            LIMIT 50
        '''

    @query('argument', 1.0, 'Customers with orders but no line items')
    def _orders_no_lineitems(self):
        return f'''
            MATCH (c:Customer)
            WHERE c.c_custkey >= {self._param_custkey()}
            AND EXISTS {{
                MATCH (c)-[:PLACED]->(:Orders)-[:HAS_ITEM]->(l:LineItem)
                WHERE l.l_quantity >= {self._param_int('quantity', 20, 30)}
            }}
            AND NOT EXISTS {{
                MATCH (c)-[:PLACED]->(:Orders)-[:HAS_ITEM]->(:LineItem)-[:SUPPLIED_BY]->(s:Supplier)
                WHERE s.s_suppkey <= {self._param_suppkey()}
            }}
            RETURN c.c_custkey
            ORDER BY c.c_custkey
            LIMIT {self._param_limit(500)}
        '''

    @query('argument', 1.0, 'Suppliers that never served certain customers')
    def _suppliers_never_customers(self):
        return f'''
            MATCH (c:Customer)
            WHERE c.c_custkey >= {self._param_custkey()}
                AND EXISTS {{
                    MATCH (c)-[:PLACED]->(o:Orders)
                    WHERE o.o_orderkey >= {self._param_orderkey()}
                }}
            RETURN c.c_custkey
            ORDER BY c.c_custkey
            LIMIT {self._param_limit(500)}
        '''

    #endregion
    #region TriadicSelection
    # Should look like this:
    # MATCH (a)--(b)--(c)
    # WHERE NOT (a)--(c)
    # This is kinda a problem because we don't have any relationship between the same two kinds.
    # But we can add one!

    @query('triadic', 1.0, 'Cross-nation recommendations')
    def _cross_nation_recommendations(self):
        return f'''
            MATCH (c1:Customer)-[:KNOWS]->(:Customer)-[:KNOWS]->(c2:Customer)
            MATCH (c1)-[:IN_NATION]->(n1:Nation)
            MATCH (c2)-[:IN_NATION]->(n2:Nation)
            WHERE NOT (c1)-[:KNOWS]->(c2)
                AND c1 <> c2
                AND n1 <> n2
            RETURN c1.c_custkey, c2.c_custkey, COUNT(*) AS score
            ORDER BY score DESC
            LIMIT {self._param_limit(500)}
        '''

    @query('triadic', 1.0, 'High-value customer network recommendations')
    def _customer_reommendations(self):
        min_orders = self._param_int('min_orders', 20, 50)

        return f'''
            MATCH (c1:Customer)-[:KNOWS]->(:Customer)-[:KNOWS]->(c2:Customer)
            MATCH (c1)-[:PLACED]->(o1:Orders)
            MATCH (c2)-[:PLACED]->(o2:Orders)
            WHERE NOT (c1)-[:KNOWS]->(c2)
                AND c1 <> c2
            WITH c1, c2, COUNT(DISTINCT o1) AS o1c, COUNT(DISTINCT o2) AS o2c
            WHERE o1c > {min_orders}
                AND o2c > {min_orders}
            RETURN c1.c_custkey, c2.c_custkey, o1c, o2c
            ORDER BY (o1c + o2c) DESC
            LIMIT 200
        '''

    @query('triadic', 1.0, 'Friend-of-friend recommendations')
    def _friend_recommendations(self):
        return '''
            MATCH (c1:Customer)-[:KNOWS]->(:Customer)-[:KNOWS]->(c2:Customer)
            WHERE NOT (c1)-[:KNOWS]->(c2)
                AND c1 <> c2
            RETURN c1.c_custkey, c2.c_custkey, COUNT(*) AS score
            ORDER BY score DESC
            LIMIT 50
        '''

    #endregion
    #region NodeUniqueIndexSeek
    # This one is pretty simple, we just have to query for a specific node by its unique property.

    @query('index-seek', 1.0, 'Customers by id')
    def _customers_by_id(self):
        return f'MATCH (n:Customer {{ c_custkey: {self._param_custkey()} }}) RETURN n'

    @query('index-seek', 1.0, 'Orders by id')
    def _orders_by_id(self):
        return f'MATCH (n:Orders {{ o_orderkey: {self._param_orderkey()} }}) RETURN n'

    @query('index-seek', 1.0, 'Suppliers by id')
    def _suppliers_by_id(self):
        return f'MATCH (n:Part {{ p_partkey: {self._param_partkey()} }}) RETURN n'

    #endregion
    #region OrderedAggregation
    # Should look like this:
    # MATCH (a)
    # ORDER BY a.prop
    # RETURN a.prop, count(*)

    @query('order-agg', 1.0, 'Revenue per nation, ordered by nation name')
    def _ordered_revenue_per_nation(self):
        # TODO param
        return f'''
            MATCH (n:Nation)<-[:IN_NATION]-(s:Supplier)<-[:SUPPLIED_BY]-(l:LineItem)
            ORDER BY n.n_name
            RETURN n.n_name, sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
        '''

    @query('order-agg', 1.0, 'Orders per customer, ordered by customer name')
    def _ordered_orders_per_customer(self):
        # TODO param
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Orders)
            ORDER BY c.c_custkey
            RETURN c.c_custkey, count(*) AS orders
        '''

    #endregion
    #region NodeIndexScan
    # It looks like that we need to scan an index but not for a specific value. E.g., we want the property to be "NOT NULL" instead.
    # We might think that something like:
    # - filter date by day of week
    # - filter int by modulo
    # would work ... but tough luck, Neo4j just doesn't care! Maybe a larger dataset would help? Who knows.
    # The only thing that seems to work is the "NOT NULL" filter so let's do that (even if all data values are not null ...).

    @query('index-scan', 1.0, 'Filter by non-null date (should use index)')
    def _orders_by_month(self):
        return f'''
            MATCH (o:Orders)
            WHERE o.o_orderdate IS NOT NULL
                AND date(o.o_orderdate).month = {self._param_month()}
            RETURN o.o_orderkey, o.o_orderdate
            LIMIT 1000
        '''

    @query('index-scan', 1.0, 'Filter by modulo on an indexed numeric property (should use index)')
    def _customers_modulo_id(self):
        return f'''
            MATCH (c:Customer)
            WHERE c.c_custkey % {self._param_int('modulo', 69, 420)} = 0
                AND c.c_custkey IS NOT NULL
            RETURN count(*) AS bucket_size
        '''

    @query('index-scan', 1.0, 'LineItems shipped on Mondays')
    def _lineitems_by_shipdate(self):
        return '''
            MATCH (l:LineItem)
            WHERE l.l_shipdate IS NOT NULL
                AND date(l.l_shipdate).dayOfWeek = 1
            RETURN l.l_orderkey, l.l_shipdate
            LIMIT 1000
        '''

    @query('index-scan', 1.0, 'PartSupp buckets (there is no deeper meaning)')
    def _partsupp_buckets(self):
        return '''
            MATCH (ps:PartSupp)
            WHERE ps.ps_partkey + ps.ps_suppkey > 10000
                AND ps.ps_partkey IS NOT NULL
                AND ps.ps_suppkey IS NOT NULL
            RETURN count(*) AS bucket_size
        '''

    #endregion
    #region Rogue operators

    @query('order', 1.0, 'Customers by nation')
    def _order_agg1(self):
        return '''
            MATCH (c:Customer)-[:IN_NATION]->(n:Nation)
            ORDER BY n.n_name
            RETURN n.n_name, count(*) AS customer_count
        '''

    @query('order', 1.0, 'Parts by usage count')
    def _order_agg2(self):
        return '''
            MATCH (p:Part)<-[:OF_PART]-(l:LineItem)
            ORDER BY p.p_partkey
            RETURN p.p_partkey, count(*) AS usage_count
        '''

    #endregion
