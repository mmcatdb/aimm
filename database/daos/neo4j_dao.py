from neo4j import GraphDatabase
from .base_dao import BaseDAO

class Neo4jDAO(BaseDAO):
    def __init__(self, config):
        self.config = config
        # self.driver = GraphDatabase.driver(config['uri'], auth=(config['user'], config['password'])) if config else None
        self.connect()

    def connect(self):
        self.driver = GraphDatabase.driver(self.config['uri'], auth=(self.config['user'], self.config['password']))

    def disconnect(self):
        self.driver.close()

    def _execute_query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def find(self, entity_name, query_params):
        conditions = []
        params = {}
        for key, value in query_params.items():
            prop = key.split('__')[0]
            if key.endswith("__in"):
                conditions.append(f"n.{prop} IN ${prop}")
                params[prop] = value
            else:
                conditions.append(f"n.{prop} = ${prop}")
                params[prop] = value
        
        where_clause = " AND ".join(conditions)
        query = f"MATCH (n:{entity_name}) WHERE {where_clause} RETURN n"
        
        results = self._execute_query(query, params)
        return [res['n'] for res in results]

    def insert(self, entity_name, data):
        query = f"CREATE (n:{entity_name} $props)"
        self._execute_query(query, {'props': data})

    def create_schema(self, entity_name, schema):
        # Could create constraints in place of a schema, but it's not necessary (at least for now)
        pass


    def delete_all_from(self, entity_name):
        query = f"MATCH (n:{entity_name}) DETACH DELETE n"
        self._execute_query(query)
        print(f"All data from '{entity_name}' has been deleted in Neo4j.")

    def drop_entity(self, entity_name):
        self.delete_all_from(entity_name)
        print(f"All nodes with label '{entity_name}' have been dropped in Neo4j.")


    # A1) Non-Indexed Columns
    def get_all_lineitems(self):
        return self._execute_query("MATCH (l:lineitem) RETURN l")

    # A2) Non-Indexed Columns — Range Query
    def get_orders_by_daterange(self, start_date, end_date):
        query = "MATCH (o:orders) WHERE o.o_orderdate >= $start_date AND o.o_orderdate <= $end_date RETURN o"
        return self._execute_query(query, {'start_date': start_date, 'end_date': end_date})

    # A3) Indexed Columns
    def get_all_customers(self):
        return self._execute_query("MATCH (c:customer) RETURN c")

    # A4) Indexed Columns — Range Query
    def get_orders_by_keyrange(self, start_key, end_key):
        query = "MATCH (o:orders) WHERE o.o_orderkey >= $start_key AND o.o_orderkey <= $end_key RETURN o"
        return self._execute_query(query, {'start_key': start_key, 'end_key': end_key})

    # B1) COUNT
    def count_orders_by_month(self):
        """Counts orders grouped by month."""
        query = """
            MATCH (o:orders)
            RETURN count(o.o_orderkey) AS order_count, 
                   substring(o.o_orderdate, 0, 7) AS order_month
            ORDER BY order_month
        """
        return self._execute_query(query)

    # B2) MAX
    def get_max_price_by_ship_month(self):
        """Finds max extended price grouped by shipping month."""
        query = """
            MATCH (l:lineitem)
            RETURN substring(l.l_shipdate, 0, 7) AS ship_month,
                   max(l.l_extendedprice) AS max_price
            ORDER BY ship_month
        """
        return self._execute_query(query)
