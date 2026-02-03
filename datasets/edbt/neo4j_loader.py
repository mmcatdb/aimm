from typing_extensions import override
from common.config import Config
from common.drivers import Neo4jDriver
from common.loaders.neo4j_loader import Neo4jLoader

class EdbtNeo4jLoader(Neo4jLoader):
    def __init__(self, config: Config, driver: Neo4jDriver):
        super().__init__(config, driver)

    @override
    def name(self) -> str:
        return 'EDBT'

    @override
    def _get_kinds(self):
        return ['category', 'follows', 'has_category', 'has_interest', 'order_item', 'order', 'product', 'review', 'seller', 'similar', 'user']

    @override
    def _define_constraints(self):
        return [
            # Unique keys
            'CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE',
            'CREATE CONSTRAINT seller_id_unique IF NOT EXISTS FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE',
            'CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE',
            'CREATE CONSTRAINT category_id_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.category_id IS UNIQUE',
            'CREATE CONSTRAINT order_id_unique IF NOT EXISTS FOR (o:Order) REQUIRE o.order_id IS UNIQUE',
            # Useful indexes
            'CREATE INDEX product_active IF NOT EXISTS FOR (p:Product) ON (p.is_active)',
            'CREATE INDEX order_buyer IF NOT EXISTS FOR (o:Order) ON (o.buyer_user_id)',
            'CREATE INDEX category_path IF NOT EXISTS FOR (c:Category) ON (c.path)',
        ]

    @override
    def _load_data(self):
        # Load nodes first

        self._load_csv('user', '''
            CREATE (:User {
                user_id: toInteger(row[0]),
                handle: row[1],
                email: row[2],
                created_at: datetime(row[3]),
                country_code: row[4],
                is_active: (row[5] = 'true'),
                profile: row[6]
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
                buyer_user_id: toInteger(row[1]),
                order_ts: datetime(row[2]),
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
                seller_id: toInteger(row[3]),
                unit_price_cents: toInteger(row[4]),
                quantity: toInteger(row[5]),
                line_total_cents: toInteger(row[6]),
                created_at: datetime(row[7]),
                product_snapshot: row[8]
            }]->(p)
        ''')

        self._load_csv('review', '''
            MATCH (p:Product {product_id: toInteger(row[1])}),
                (u:User {user_id: toInteger(row[2])})
            CREATE (u)-[:REVIEWED {
                review_id: toInteger(row[0]),
                product_id: toInteger(row[1]),
                user_id: toInteger(row[2]),
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
            MATCH (u:User {user_id: toInteger(row[0])}),
                (c:Category {category_id: toInteger(row[1])})
            CREATE (u)-[:HAS_INTEREST {
                strength: toInteger(row[2]),
                created_at: datetime(row[3])
            }]->(c)
        ''')

        self._load_csv('follows', '''
            MATCH (a:User {user_id: toInteger(row[0])}),
                (b:User {user_id: toInteger(row[1])})
            CREATE (a)-[:FOLLOWS {
                created_at: datetime(row[2])
            }]->(b)
        ''')

        # Create relationships for simple foreign keys

        print("Creating Seller -> Product relationships...")
        self._dao.execute("""
            MATCH (p:Product), (s:Seller {seller_id: p.seller_id})
            CREATE (s)-[:OFFERS]->(p)
        """)
        self._dao.execute("MATCH (p:Product) REMOVE p.seller_id")

        print("Creating User -> Order relationships...")
        self._dao.execute("""
            MATCH (o:Order), (u:User {user_id: o.buyer_user_id})
            CREATE (u)-[:PLACED]->(o)
        """)
        self._dao.execute("MATCH (o:Order) REMOVE o.buyer_user_id")
