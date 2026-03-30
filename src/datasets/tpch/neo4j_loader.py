from typing_extensions import override
from common.loaders.neo4j_loader import Neo4jLoader
from common.config import DatasetName

class TpchNeo4jLoader(Neo4jLoader):

    @override
    def dataset(self):
        return DatasetName.TPCH

    @override
    def _get_kinds(self):
        return ['region', 'nation', 'part', 'supplier', 'customer', 'orders', 'partsupp', 'lineitem', 'knows']

    @override
    def _define_constraints(self):
        return [
            'CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.r_regionkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (n:Nation) REQUIRE n.n_nationkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.c_custkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (o:Orders) REQUIRE o.o_orderkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.p_partkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.s_suppkey IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (ps:PartSupp) REQUIRE (ps.ps_partkey, ps.ps_suppkey) IS UNIQUE',
            'CREATE CONSTRAINT IF NOT EXISTS FOR (li:LineItem) REQUIRE (li.l_orderkey, li.l_linenumber) IS UNIQUE',
            # An unique constraint on the KNOWS relationship is probably not legal. At this point, we should stop caring so much ...
            'CREATE INDEX IF NOT EXISTS FOR (o:Orders) ON (o.o_orderdate)',
            'CREATE INDEX IF NOT EXISTS FOR (li:LineItem) ON (li.l_shipdate)',
        ]

    @override
    def _load_data(self):
        # Load nodes first

        self._load_csv('Region', 'region', '''
            CREATE (:Region {
                r_regionkey: toInteger(row[0]),
                r_name: row[1],
                r_comment: row[2]
            })
        ''')

        self._load_csv('Nation', 'nation', '''
            CREATE (:Nation {
                n_nationkey: toInteger(row[0]),
                n_name: row[1],
                n_regionkey: toInteger(row[2]),
                n_comment: row[3]
            })
        ''')

        self._load_csv('Part', 'part', '''
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

        self._load_csv('Supplier', 'supplier', '''
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

        self._load_csv('Customer', 'customer', '''
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

        self._load_csv('Orders', 'orders', '''
            CREATE (:Orders {
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

        self._load_csv('FOR_PART', 'partsupp', '''
            MATCH (p:Part {p_partkey: toInteger(row[0])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[1])})
            CREATE (p)<-[:FOR_PART]-(ps:PartSupp {
                ps_partkey: toInteger(row[0]),
                ps_suppkey: toInteger(row[1]),
                ps_availqty: toInteger(row[2]),
                ps_supplycost: toFloat(row[3]),
                ps_comment: row[4]
            })-[:SUPPLIED_BY]->(s)
        ''', 'creating relationships to Part and Supplier')

        self._load_csv('HAS_ITEM', 'lineitem', '''
            MATCH (o:Orders {o_orderkey: toInteger(row[0])})
            MATCH (p:Part {p_partkey: toInteger(row[1])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[2])})
            CREATE (o)-[:HAS_ITEM]->(li:LineItem {
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
            })-[:OF_PART]->(p)
            CREATE (li)-[:SUPPLIED_BY]->(s)
        ''', 'creating relationships to Orders, Part and Supplier')

        # Create relationships for simple foreign keys

        self._create_relationship('IN_REGION', 'Nation',   'n_regionkey', 'Region', 'r_regionkey')
        self._create_relationship('IN_NATION', 'Customer', 'c_nationkey', 'Nation', 'n_nationkey')
        self._create_relationship('IN_NATION', 'Supplier', 's_nationkey', 'Nation', 'n_nationkey')
        self._create_relationship('PLACED',    'Customer', 'c_custkey',   'Orders', 'o_custkey')

        # Custom tables (not part of the original TPC-H schema)

        self._load_csv('KNOWS', 'knows', '''
            MATCH (c1:Customer {c_custkey: toInteger(row[0])})
            MATCH (c2:Customer {c_custkey: toInteger(row[1])})
            CREATE (c1)-[:KNOWS {
                k_startdate: date(row[2]),
                k_source: row[3],
                k_comment: row[4],
                k_strength: toFloat(row[5])
            }]->(c2)
        ''')
