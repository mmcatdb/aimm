from common.drivers import Neo4jDriver, cypher
from .base_dao import BaseDAO

class Neo4jDAO(BaseDAO):
    def __init__(self, neo4j: Neo4jDriver):
        self.neo4j = neo4j

    def _execute_query(self, query: str, params=None):
        with self.neo4j.session() as session:
            result = session.run(cypher(query), params)
            return [record.data() for record in result]

    def find(self, entity_name, query_params):
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

        query = f'MATCH (n:{entity_name}) WHERE {where_clause} RETURN n'

        results = self._execute_query(query, params)
        return [res['n'] for res in results]

    def insert(self, entity_name, data):
        query = f'CREATE (n:{entity_name} $props)'
        self._execute_query(query, {'props': data})

    def create_schema(self, entity_name, schema):
        # Could create constraints in place of a schema, but it's not necessary (at least for now)
        pass


    def delete_all_from(self, entity_name):
        query = f'MATCH (n:{entity_name}) DETACH DELETE n'
        self._execute_query(query)
        print(f'All data from "{entity_name}" has been deleted in Neo4j.')

    def drop_entity(self, entity_name):
        self.delete_all_from(entity_name)
        print(f'All nodes with label "{entity_name}" have been dropped in Neo4j.')


    def get_all_lineitems(self):
        # Neo4j really struggles with retrieving all lineitems, so limit it to 200k for testing purposes
        return self._execute_query('MATCH (l:lineitem) RETURN l LIMIT 200000')

    def get_orders_by_daterange(self, start_date, end_date):
        query = 'MATCH (o:orders) WHERE o.o_orderdate >= $start_date AND o.o_orderdate <= $end_date RETURN o'
        return self._execute_query(query, {'start_date': start_date, 'end_date': end_date})

    def get_all_customers(self):
        return self._execute_query('MATCH (c:customer) RETURN c')

    def get_orders_by_keyrange(self, start_key, end_key):
        query = 'MATCH (o:orders) WHERE o.o_orderkey >= $start_key AND o.o_orderkey <= $end_key RETURN o'
        return self._execute_query(query, {'start_key': start_key, 'end_key': end_key})

    def count_orders_by_month(self):
        query = """
            MATCH (o:orders)
            RETURN count(o.o_orderkey) AS order_count,
                   substring(o.o_orderdate, 0, 7) AS order_month
            ORDER BY order_month
        """
        return self._execute_query(query)

    def get_max_price_by_ship_month(self):
        query = """
            MATCH (l:lineitem)
            RETURN substring(l.l_shipdate, 0, 7) AS ship_month,
                   max(l.l_extendedprice) AS max_price
            ORDER BY ship_month
        """
        return self._execute_query(query)

    # --- Part / Supplier / PartSupp ---
    def get_all_parts(self):
        return self._execute_query('MATCH (p:part) RETURN p')

    def get_parts_by_size_range(self, min_size, max_size):
        query = 'MATCH (p:part) WHERE toInteger(p.p_size) >= $min AND toInteger(p.p_size) <= $max RETURN p'
        return self._execute_query(query, {'min': int(min_size), 'max': int(max_size)})

    def get_all_suppliers(self):
        return self._execute_query('MATCH (s:supplier) RETURN s')

    def get_suppliers_by_nation(self, nation_key):
        query = 'MATCH (s:supplier) WHERE s.s_nationkey = $nk RETURN s'
        return self._execute_query(query, {'nk': str(nation_key)})

    def get_partsupp_for_part(self, partkey):
        query = 'MATCH (ps:partsupp) WHERE ps.ps_partkey = $pk RETURN ps'
        return self._execute_query(query, {'pk': str(partkey)})

    def get_lowest_cost_supplier_for_part(self, partkey):
        query = """
            MATCH (ps:partsupp {ps_partkey: $pk})
            WITH ps ORDER BY toFloat(ps.ps_supplycost) ASC LIMIT 1
            MATCH (s:supplier {s_suppkey: ps.ps_suppkey})
            RETURN ps, s.s_name AS s_name, s.s_acctbal AS s_acctbal
        """
        res = self._execute_query(query, {'pk': str(partkey)})
        return res[0] if res else None

    def count_suppliers_per_part(self):
        query = """
            MATCH (ps:partsupp)
            WITH ps.ps_partkey AS partkey, count(ps) AS supplier_count
            RETURN partkey, supplier_count
            ORDER BY partkey
        """
        return self._execute_query(query)

    def avg_supplycost_by_part_size(self):
        query = """
            MATCH (p:part)<-[:PART_REL]- (ps:partsupp)
            WITH toInteger(p.p_size) AS p_size, avg(toFloat(ps.ps_supplycost)) AS avg_supplycost
            RETURN p_size, avg_supplycost
            ORDER BY p_size
        """
        return self._execute_query(query)
