from typing_extensions import override
from core.loaders.neo4j_loader import Neo4jLoader

def export():
    return Neo4jTpchLoader()

class Neo4jTpchLoader(Neo4jLoader):

    @override
    def _get_kinds(self):
        return ['region', 'nation', 'part', 'supplier', 'customer', 'orders', 'partsupp', 'lineitem', 'knows']

    @override
    def _get_constraints(self):
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
                r_regionkey: toInteger(row.r_regionkey),
                r_name: row.r_name,
                r_comment: row.r_comment
            })
        ''')

        self._load_csv('Nation', 'nation', '''
            CREATE (:Nation {
                n_nationkey: toInteger(row.n_nationkey),
                n_name: row.n_name,
                n_regionkey: toInteger(row.n_regionkey),
                n_comment: row.n_comment
            })
        ''')

        self._load_csv('Part', 'part', '''
            CREATE (:Part {
                p_partkey: toInteger(row.p_partkey),
                p_name: row.p_name,
                p_mfgr: row.p_mfgr,
                p_brand: row.p_brand,
                p_type: row.p_type,
                p_size: toInteger(row.p_size),
                p_container: row.p_container,
                p_retailprice: toFloat(row.p_retailprice),
                p_comment: row.p_comment
            })
        ''')

        self._load_csv('Supplier', 'supplier', '''
            CREATE (:Supplier {
                s_suppkey: toInteger(row.s_suppkey),
                s_name: row.s_name,
                s_address: row.s_address,
                s_nationkey: toInteger(row.s_nationkey),
                s_phone: row.s_phone,
                s_acctbal: toFloat(row.s_acctbal),
                s_comment: row.s_comment
            })
        ''')

        self._load_csv('Customer', 'customer', '''
            CREATE (:Customer {
                c_custkey: toInteger(row.c_custkey),
                c_name: row.c_name,
                c_address: row.c_address,
                c_nationkey: toInteger(row.c_nationkey),
                c_phone: row.c_phone,
                c_acctbal: toFloat(row.c_acctbal),
                c_mktsegment: row.c_mktsegment,
                c_comment: row.c_comment
            })
        ''')

        self._load_csv('Orders', 'orders', '''
            CREATE (:Orders {
                o_orderkey: toInteger(row.o_orderkey),
                o_custkey: toInteger(row.o_custkey),
                o_orderstatus: row.o_orderstatus,
                o_totalprice: toFloat(row.o_totalprice),
                o_orderdate: date(row.o_orderdate),
                o_orderpriority: row.o_orderpriority,
                o_clerk: row.o_clerk,
                o_shippriority: toInteger(row.o_shippriority),
                o_comment: row.o_comment
            })
        ''')

        # Load nodes and relationships for many-to-many tables

        self._load_csv('FOR_PART', 'partsupp', '''
            MATCH (p:Part {p_partkey: toInteger(row.ps_partkey)})
            MATCH (s:Supplier {s_suppkey: toInteger(row.ps_suppkey)})
            CREATE (p)<-[:FOR_PART]-(ps:PartSupp {
                ps_partkey: toInteger(row.ps_partkey),
                ps_suppkey: toInteger(row.ps_suppkey),
                ps_availqty: toInteger(row.ps_availqty),
                ps_supplycost: toFloat(row.ps_supplycost),
                ps_comment: row.ps_comment
            })-[:SUPPLIED_BY]->(s)
        ''', 'creating relationships to Part and Supplier')

        self._load_csv('HAS_ITEM', 'lineitem', '''
            MATCH (o:Orders {o_orderkey: toInteger(row.l_orderkey)})
            MATCH (p:Part {p_partkey: toInteger(row.l_partkey)})
            MATCH (s:Supplier {s_suppkey: toInteger(row.l_suppkey)})
            CREATE (o)-[:HAS_ITEM]->(li:LineItem {
                l_orderkey: toInteger(row.l_orderkey),
                l_partkey: toInteger(row.l_partkey),
                l_suppkey: toInteger(row.l_suppkey),
                l_linenumber: toInteger(row.l_linenumber),
                l_quantity: toFloat(row.l_quantity),
                l_extendedprice: toFloat(row.l_extendedprice),
                l_discount: toFloat(row.l_discount),
                l_tax: toFloat(row.l_tax),
                l_returnflag: row.l_returnflag,
                l_linestatus: row.l_linestatus,
                l_shipdate: date(row.l_shipdate),
                l_commitdate: date(row.l_commitdate),
                l_receiptdate: date(row.l_receiptdate),
                l_shipinstruct: row.l_shipinstruct,
                l_shipmode: row.l_shipmode,
                l_comment: row.l_comment
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
            MATCH (c1:Customer {c_custkey: toInteger(row.k_custkey1)})
            MATCH (c2:Customer {c_custkey: toInteger(row.k_custkey2)})
            CREATE (c1)-[:KNOWS {
                k_startdate: date(row.k_startdate),
                k_source: row.k_source,
                k_comment: row.k_comment,
                k_strength: toFloat(row.k_strength)
            }]->(c2)
        ''')
