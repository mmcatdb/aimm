from typing_extensions import override
from common.config import Config
from common.drivers import Neo4jDriver
from common.loaders.neo4j_loader import Neo4jLoader

class TpchNeo4jLoader(Neo4jLoader):
    def __init__(self, config: Config, driver: Neo4jDriver):
        super().__init__(config, driver)

    @override
    def name(self) -> str:
        return 'TPC-H'

    @override
    def _get_kinds(self):
        return ['region', 'nation', 'part', 'supplier', 'customer', 'orders', 'partsupp', 'lineitem']

    @override
    def _define_constraints(self):
        return [
            'CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.r_regionkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (n:Nation) REQUIRE n.n_nationkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.c_custkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (o:Order) REQUIRE o.o_orderkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.p_partkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.s_suppkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (ps:PartSupp) REQUIRE (ps.ps_partkey, ps.ps_suppkey) IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (li:LineItem) REQUIRE (li.l_orderkey, li.l_linenumber) IS UNIQUE',
        ]

    @override
    def _load_data(self):
        # Load nodes first

        self._load_csv('region', '''
            CREATE (:Region {
                r_regionkey: toInteger(row[0]),
                r_name: row[1],
                r_comment: row[2]
            })
        ''')

        self._load_csv('nation', '''
            CREATE (:Nation {
                n_nationkey: toInteger(row[0]),
                n_name: row[1],
                n_regionkey: toInteger(row[2]),
                n_comment: row[3]
            })
        ''')

        self._load_csv('part', '''
            CREATE (:Part {
                p_partkey: toInteger(row[0]),
                p_name: row[1],
                p_mfgr: row[2],
                p_brand: row[3],
                p_type: row[4],
                p_size: toInteger(row[5]),
                p_container: row[6],
                p_retailprice: toFloat(row[7]),
                p_comment: row[8]
            })
        ''')

        self._load_csv('supplier', '''
            CREATE (:Supplier {
                s_suppkey: toInteger(row[0]),
                s_name: row[1],
                s_address: row[2],
                s_nationkey: toInteger(row[3]),
                s_phone: row[4],
                s_acctbal: toFloat(row[5]),
                s_comment: row[6]
            })
        ''')

        self._load_csv('customer', '''
            CREATE (:Customer {
                c_custkey: toInteger(row[0]),
                c_name: row[1],
                c_address: row[2],
                c_nationkey: toInteger(row[3]),
                c_phone: row[4],
                c_acctbal: toFloat(row[5]),
                c_mktsegment: row[6],
                c_comment: row[7]
            })
        ''')

        self._load_csv('orders', '''
            CREATE (:Order {
                o_orderkey: toInteger(row[0]),
                o_custkey: toInteger(row[1]),
                o_orderstatus: row[2],
                o_totalprice: toFloat(row[3]),
                o_orderdate: date(row[4]),
                o_orderpriority: row[5],
                o_clerk: row[6],
                o_shippriority: toInteger(row[7]),
                o_comment: row[8]
            })
        ''')

        # Load nodes and relationships for many-to-many tables

        self._load_csv('partsupp', '''
            MATCH (p:Part {p_partkey: toInteger(row[0])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[1])})
            CREATE (p)<-[:IS_FOR_PART]-(ps:PartSupp {
                ps_partkey: toInteger(row[0]),
                ps_suppkey: toInteger(row[1]),
                ps_availqty: toInteger(row[2]),
                ps_supplycost: toFloat(row[3]),
                ps_comment: row[4]
            })-[:SUPPLIED_BY]->(s)
        ''', 'creating relationships to Part and Supplier')

        self._load_csv('lineitem', '''
            MATCH (o:Order {o_orderkey: toInteger(row[0])})
            MATCH (p:Part {p_partkey: toInteger(row[1])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[2])})
            MATCH (p)<-[:IS_FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s)
            CREATE (o)-[:CONTAINS_ITEM]->(li:LineItem {
                l_orderkey: toInteger(row[0]),
                l_partkey: toInteger(row[1]),
                l_suppkey: toInteger(row[2]),
                l_linenumber: toInteger(row[3]),
                l_quantity: toFloat(row[4]),
                l_extendedprice: toFloat(row[5]),
                l_discount: toFloat(row[6]),
                l_tax: toFloat(row[7]),
                l_returnflag: row[8],
                l_linestatus: row[9],
                l_shipdate: date(row[10]),
                l_commitdate: date(row[11]),
                l_receiptdate: date(row[12]),
                l_shipinstruct: row[13],
                l_shipmode: row[14],
                l_comment: row[15]
            })-[:IS_PRODUCT_SUPPLY]->(ps)
        ''', 'creating relationships to Order and PartSupp')

        # Create relationships for simple foreign keys

        print('Creating Nation -> Region relationships...')
        self._dao.execute('''
            MATCH (n:Nation), (r:Region {r_regionkey: n.n_regionkey})
            CREATE (n)-[:IS_IN_REGION]->(r)
        ''')
        # Remove redundant foreign key property
        self._dao.execute('MATCH (n:Nation) REMOVE n.n_regionkey')

        print('Creating Customer -> Nation relationships...')
        self._dao.execute('''
            MATCH (c:Customer), (n:Nation {n_nationkey: c.c_nationkey})
            CREATE (c)-[:IS_IN_NATION]->(n)
        ''')
        self._dao.execute('MATCH (c:Customer) REMOVE c.c_nationkey')

        print('Creating Supplier -> Nation relationships...')
        self._dao.execute('''
            MATCH (s:Supplier), (n:Nation {n_nationkey: s.s_nationkey})
            CREATE (s)-[:IS_IN_NATION]->(n)
        ''')
        self._dao.execute('MATCH (s:Supplier) REMOVE s.s_nationkey')

        print('Creating Customer -> Order relationships...')
        self._dao.execute('''
            MATCH (c:Customer), (o:Order {o_custkey: c.c_custkey})
            CREATE (c)-[:PLACED]->(o)
        ''')
        self._dao.execute('MATCH (o:Order) REMOVE o.o_custkey')
