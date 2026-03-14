from typing_extensions import override
from common.loaders.neo4j_loader import Neo4jLoader

# TODO now we add foreign keys in neo4j (and then we remove them for some kinds)
# - they are probably used in some queries (in TPC-H) but they shouldn't (and after that, we should remove them)

class EdbtNeo4jLoader(Neo4jLoader):
    @override
    def name(self):
        return 'EDBT'

    @override
    def _get_kinds(self):
        return ['category', 'follows', 'has_category', 'has_interest', 'order_item', 'order', 'product', 'review', 'seller', 'customer', 'person']

    @override
    def _define_constraints(self):
        return [
            # Unique keys
            'CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE',
            'CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE',
            'CREATE CONSTRAINT seller_id_unique IF NOT EXISTS FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE',
            'CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE',
            'CREATE CONSTRAINT category_id_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.category_id IS UNIQUE',
            'CREATE CONSTRAINT order_id_unique IF NOT EXISTS FOR (o:Order) REQUIRE o.order_id IS UNIQUE',
            # Useful indexes
            'CREATE INDEX product_active IF NOT EXISTS FOR (p:Product) ON (p.is_active)',
            'CREATE INDEX order_customer IF NOT EXISTS FOR (o:Order) ON (o.customer_id)',
            'CREATE INDEX category_path IF NOT EXISTS FOR (c:Category) ON (c.path)',
        ]

    @override
    def _load_data(self):
        # Load nodes first

        self._load_csv('person', '''
            CREATE (:Person {
                person_id: toInteger(row[0]),
                name: row[1],
                email: row[2],
                created_at: datetime(row[3]),
                country_code: row[4],
                is_active: (row[5] = 'true'),
                profile: row[6]
            })
        ''')

        self._load_csv('customer', '''
            CREATE (:Customer {
                customer_id: toInteger(row[0]),
                person_id: toInteger(row[1]),
                snapshot_at: datetime(row[2]),
                name: row[3],
                email: row[4],
                country_code: row[5],
                is_active: (row[6] = 'true'),
                profile: row[7]
            })
        ''')

        self._load_csv('seller', '''
            CREATE (:Seller {
                seller_id: toInteger(row[0]),
                display_name: row[1],
                created_at: datetime(row[2]),
                country_code: row[3],
                is_active: (row[4] = 'true')
            })
        ''')

        self._load_csv('category', '''
            CREATE (:Category {
                category_id: toInteger(row[0]),
                name: row[1],
                path: row[2]
            })
        ''')

        self._load_csv('product', '''
            CREATE (:Product {
                product_id: toInteger(row[0]),
                seller_id: toInteger(row[1]),
                sku: row[2],
                title: row[3],
                description: row[4],
                price_cents: toInteger(row[5]),
                currency: row[6],
                stock_qty: toInteger(row[7]),
                is_active: (row[8] = 'true'),
                created_at: datetime(row[9]),
                updated_at: datetime(row[10]),
                attributes: row[11]
            })
        ''')

        self._load_csv('order', '''
            CREATE (:Order {
                order_id: toInteger(row[0]),
                customer_id: toInteger(row[1]),
                ordered_at: datetime(row[2]),
                status: row[3],
                total_cents: toInteger(row[4]),
                currency: row[5],
                shipping: row[6],
                payment: row[7]
            })
        ''')

        # Load nodes and relationships for many-to-many tables

        self._load_csv('order_item', '''
            MATCH (o:Order {order_id: toInteger(row[1])}),
                (p:Product {product_id: toInteger(row[2])})
            CREATE (o)-[:HAS_ITEM {
                order_item_id: toInteger(row[0]),
                unit_price_cents: toInteger(row[3]),
                quantity: toInteger(row[4]),
                line_total_cents: toInteger(row[5]),
                created_at: datetime(row[6]),
                product_snapshot: row[7]
            }]->(p)
        ''')

        self._load_csv('review', '''
            MATCH (p:Product {product_id: toInteger(row[1])}),
                (u:Customer {customer_id: toInteger(row[2])})
            CREATE (u)-[:REVIEWED {
                review_id: toInteger(row[0]),
                product_id: toInteger(row[1]),
                customer_id: toInteger(row[2]),
                rating: toInteger(row[3]),
                title: row[4],
                body: row[5],
                created_at: datetime(row[6]),
                helpful_votes: toInteger(row[7])
            }]->(p)
        ''')

        self._load_csv('has_category', '''
            MATCH (p:Product {product_id: toInteger(row[0])}),
                (c:Category {category_id: toInteger(row[1])})
            CREATE (p)-[:HAS_CATEGORY {
                assigned_at: datetime(row[2])
            }]->(c)
        ''')

        self._load_csv('has_interest', '''
            MATCH (u:Person {person_id: toInteger(row[0])}),
                (c:Category {category_id: toInteger(row[1])})
            CREATE (u)-[:HAS_INTEREST {
                strength: toInteger(row[2]),
                created_at: datetime(row[3])
            }]->(c)
        ''')

        self._load_csv('follows', '''
            MATCH (a:Person {person_id: toInteger(row[0])}),
                (b:Person {person_id: toInteger(row[1])})
            CREATE (a)-[:FOLLOWS {
                created_at: datetime(row[2])
            }]->(b)
        ''')

        # Create relationships for simple foreign keys

        print("Creating Customer -> Person relationships...")
        self._dao.execute("""
        MATCH (c:Customer), (p:Person {person_id: c.person_id})
        CREATE (c)-[:SNAPSHOT_OF]->(p)
        """)
        self._dao.execute("MATCH (c:Customer) REMOVE c.person_id")

        print("Creating Seller -> Product relationships...")
        self._dao.execute("""
            MATCH (p:Product), (s:Seller {seller_id: p.seller_id})
            CREATE (s)-[:OFFERS]->(p)
        """)
        self._dao.execute("MATCH (p:Product) REMOVE p.seller_id")

        print("Creating Customer -> Order relationships...")
        self._dao.execute("""
            MATCH (o:Order), (u:Customer {customer_id: o.customer_id})
            CREATE (u)-[:PLACED]->(o)
        """)
        self._dao.execute("MATCH (o:Order) REMOVE o.customer_id")
