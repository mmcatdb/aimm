from typing_extensions import override
from common.drivers import Neo4jDriver
from datasets.tpch.daos.tpch_dao import TpchDAO

class TpchNeo4jDAO(TpchDAO):
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    @override
    def find(self, entity: str, query_params) -> list[dict]:
        conditions = []
        params = {}
        for key, value in query_params.items():
            prop = key.split('__')[0]
            if key.endswith('__in'):
                conditions.append(f'n.{prop} IN ${prop}')
                params[prop] = value
            else:
                conditions.append(f'n.{prop} = ${prop}')
                params[prop] = value


        where_clause = ' AND '.join(conditions)

        query = f'MATCH (n:{entity}) WHERE {where_clause} RETURN n'

        results = self.driver.execute(query, params)
        return [res['n'] for res in results]

    @override
    def get_all_lineitems(self):
        # Neo4j really struggles with retrieving all lineitems, so limit it to 200k for testing purposes
        return self.driver.execute('MATCH (l:lineitem) RETURN l LIMIT 200000')

    @override
    def get_orders_by_daterange(self, start_date, end_date):
        query = 'MATCH (o:orders) WHERE o.o_orderdate >= $start_date AND o.o_orderdate <= $end_date RETURN o'
        return self.driver.execute(query, {'start_date': start_date, 'end_date': end_date})

    @override
    def get_all_customers(self):
        return self.driver.execute('MATCH (c:customer) RETURN c')

    @override
    def get_orders_by_keyrange(self, start_key, end_key):
        query = 'MATCH (o:orders) WHERE o.o_orderkey >= $start_key AND o.o_orderkey <= $end_key RETURN o'
        return self.driver.execute(query, {'start_key': start_key, 'end_key': end_key})

    @override
    def count_orders_by_month(self):
        query = """
            MATCH (o:orders)
            RETURN count(o.o_orderkey) AS order_count,
                   substring(o.o_orderdate, 0, 7) AS order_month
            ORDER BY order_month
        """
        return self.driver.execute(query)

    @override
    def get_max_price_by_ship_month(self):
        query = """
            MATCH (l:lineitem)
            RETURN substring(l.l_shipdate, 0, 7) AS ship_month,
                   max(l.l_extendedprice) AS max_price
            ORDER BY ship_month
        """
        return self.driver.execute(query)

    # --- Part / Supplier / PartSupp ---
    @override
    def get_all_parts(self):
        return self.driver.execute('MATCH (p:part) RETURN p')

    @override
    def get_parts_by_size_range(self, min_size, max_size):
        query = 'MATCH (p:part) WHERE toInteger(p.p_size) >= $min AND toInteger(p.p_size) <= $max RETURN p'
        return self.driver.execute(query, {'min': int(min_size), 'max': int(max_size)})

    @override
    def get_all_suppliers(self):
        return self.driver.execute('MATCH (s:supplier) RETURN s')

    @override
    def get_suppliers_by_nation(self, nation_key):
        query = 'MATCH (s:supplier) WHERE s.s_nationkey = $nk RETURN s'
        return self.driver.execute(query, {'nk': str(nation_key)})

    @override
    def get_partsupp_for_part(self, partkey):
        query = 'MATCH (ps:partsupp) WHERE ps.ps_partkey = $pk RETURN ps'
        return self.driver.execute(query, {'pk': str(partkey)})

    @override
    def get_lowest_cost_supplier_for_part(self, partkey):
        query = """
            MATCH (ps:partsupp {ps_partkey: $pk})
            WITH ps ORDER BY toFloat(ps.ps_supplycost) ASC LIMIT 1
            MATCH (s:supplier {s_suppkey: ps.ps_suppkey})
            RETURN ps, s.s_name AS s_name, s.s_acctbal AS s_acctbal
        """
        res = self.driver.execute(query, {'pk': str(partkey)})
        return res[0] if res else None

    @override
    def count_suppliers_per_part(self):
        query = """
            MATCH (ps:partsupp)
            WITH ps.ps_partkey AS partkey, count(ps) AS supplier_count
            RETURN partkey, supplier_count
            ORDER BY partkey
        """
        return self.driver.execute(query)

    @override
    def avg_supplycost_by_part_size(self):
        query = """
            MATCH (p:part)<-[:PART_REL]- (ps:partsupp)
            WITH toInteger(p.p_size) AS p_size, avg(toFloat(ps.ps_supplycost)) AS avg_supplycost
            RETURN p_size, avg_supplycost
            ORDER BY p_size
        """
        return self.driver.execute(query)
