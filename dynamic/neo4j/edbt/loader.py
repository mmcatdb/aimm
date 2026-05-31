from typing_extensions import override
from core.loaders.neo4j_loader import Neo4jLoader

def export():
    return Neo4jEdbtLoader()

# TODO now we add foreign keys in neo4j (and then we remove them for some kinds)
# - they are probably used in some queries (in TPC-H) but they shouldn't (and after that, we should remove them)

class Neo4jEdbtLoader(Neo4jLoader):

    @override
    def _get_kinds(self):
        return ['category', 'follows', 'has_category', 'has_interest', 'order_item', 'order', 'product', 'review', 'seller', 'customer', 'person']

    @override
    def _get_constraints(self):
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

        self._load_csv('Person', 'person', '''
            CREATE (:Person {
                person_id: toInteger(row.person_id),
                name: row.name,
                email: row.email,
                created_at: datetime(row.created_at),
                country_code: row.country_code,
                is_active: (row.is_active = 'true'),
                profile: row.profile
            })
        ''')

        self._load_csv('Customer', 'customer', '''
            CREATE (:Customer {
                customer_id: toInteger(row.customer_id),
                person_id: toInteger(row.person_id),
                snapshot_at: datetime(row.snapshot_at),
                name: row.name,
                email: row.email,
                country_code: row.country_code,
                is_active: (row.is_active = 'true'),
                profile: row.profile
            })
        ''')

        self._load_csv('Seller', 'seller', '''
            CREATE (:Seller {
                seller_id: toInteger(row.seller_id),
                display_name: row.display_name,
                created_at: datetime(row.created_at),
                country_code: row.country_code,
                is_active: (row.is_active = 'true')
            })
        ''')

        self._load_csv('Category', 'category', '''
            CREATE (:Category {
                category_id: toInteger(row.category_id),
                name: row.name,
                path: row.path
            })
        ''')

        self._load_csv('Product', 'product', '''
            CREATE (:Product {
                product_id: toInteger(row.product_id),
                seller_id: toInteger(row.seller_id),
                sku: row.sku,
                title: row.title,
                description: row.description,
                price_cents: toInteger(row.price_cents),
                currency: row.currency,
                stock_qty: toInteger(row.stock_qty),
                is_active: (row.is_active = 'true'),
                created_at: datetime(row.created_at),
                updated_at: datetime(row.updated_at),
                attributes: row.attributes
            })
        ''')

        self._load_csv('Order', 'order', '''
            CREATE (:Order {
                order_id: toInteger(row.order_id),
                customer_id: toInteger(row.customer_id),
                ordered_at: datetime(row.ordered_at),
                status: row.status,
                total_cents: toInteger(row.total_cents),
                currency: row.currency,
                shipping: row.shipping,
                payment: row.payment
            })
        ''')

        # Load nodes and relationships for many-to-many tables

        self._load_csv('HAS_ITEM', 'order_item', '''
            MATCH (o:Order {order_id: toInteger(row.order_id)}),
                (p:Product {product_id: toInteger(row.product_id)})
            CREATE (o)-[:HAS_ITEM {
                order_item_id: toInteger(row.order_item_id),
                unit_price_cents: toInteger(row.unit_price_cents),
                quantity: toInteger(row.quantity),
                line_total_cents: toInteger(row.line_total_cents),
                created_at: datetime(row.created_at)
            }]->(p)
        ''')

        self._load_csv('REVIEWED', 'review', '''
            MATCH (p:Product {product_id: toInteger(row.product_id)}),
                (u:Customer {customer_id: toInteger(row.customer_id)})
            CREATE (u)-[:REVIEWED {
                review_id: toInteger(row.review_id),
                product_id: toInteger(row.product_id),
                customer_id: toInteger(row.customer_id),
                rating: toInteger(row.rating),
                title: row.title,
                body: row.body,
                created_at: datetime(row.created_at),
                helpful_votes: toInteger(row.helpful_votes)
            }]->(p)
        ''')

        self._load_csv('HAS_CATEGORY', 'has_category', '''
            MATCH (p:Product {product_id: toInteger(row.product_id)}),
                (c:Category {category_id: toInteger(row.category_id)})
            CREATE (p)-[:HAS_CATEGORY {
                assigned_at: datetime(row.assigned_at)
            }]->(c)
        ''')

        self._load_csv('HAS_INTEREST', 'has_interest', '''
            MATCH (u:Person {person_id: toInteger(row.person_id)}),
                (c:Category {category_id: toInteger(row.category_id)})
            CREATE (u)-[:HAS_INTEREST {
                strength: toInteger(row.strength),
                created_at: datetime(row.created_at)
            }]->(c)
        ''')

        self._load_csv('FOLLOWS', 'follows', '''
            MATCH (a:Person {person_id: toInteger(row.from_id)}),
                (b:Person {person_id: toInteger(row.to_id)})
            CREATE (a)-[:FOLLOWS {
                created_at: datetime(row.created_at)
            }]->(b)
        ''')

        # Create relationships for simple foreign keys

        self._create_relationship('SNAPSHOT_OF', 'Customer', 'person_id',   'Person',  'person_id')
        self._create_relationship('OFFERS',      'Seller',   'seller_id',   'Product', 'seller_id')
        self._create_relationship('PLACED',      'Customer', 'customer_id', 'Order',   'customer_id')
