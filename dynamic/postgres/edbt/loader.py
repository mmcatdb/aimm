from typing_extensions import override
from core.loaders.postgres_loader import PostgresLoader, PostgresColumn, PostgresIndex

def export():
    return PostgresEdbtLoader()

# TODO review is for customer_id, not person_id
# TODO order_item should not have seller_id (can get from product)

class PostgresEdbtLoader(PostgresLoader):

    @override
    def _get_kinds(self):
        return get_postgres_edbt_kinds()

    @override
    def _get_constraints(self):
        return [
            PostgresIndex('person', [ 'email' ], is_unique=True),
            PostgresIndex('product', [ 'sku' ], is_unique=True),
            PostgresIndex('product', [ 'seller_id' ]),
            PostgresIndex('product', [ 'is_active' ], where='"is_active" = TRUE'),
            PostgresIndex('order', [ 'customer_id', 'ordered_at' ]),
            # ordered_at DESC
            PostgresIndex('order', [ 'ordered_at' ]),
            # ordered_at DESC
            PostgresIndex('order_item', [ 'order_id' ]),
            PostgresIndex('order_item', [ 'product_id' ]),
            PostgresIndex('review', [ 'product_id', 'customer_id' ], is_unique=True),
            PostgresIndex('review', [ 'product_id', 'created_at' ]),
            # created_at DESC
            PostgresIndex('review', [ 'customer_id', 'created_at' ]),
            # created_at DESC
            PostgresIndex('category', [ 'path' ], is_unique=True),
            PostgresIndex('has_category', [ 'category_id' ]),
            PostgresIndex('has_interest', [ 'category_id' ]),
            PostgresIndex('follows', [ 'to_id' ]),
        ]

def get_postgres_edbt_kinds() -> dict[str, list[PostgresColumn]]:
    person = [
        PostgresColumn('person_id', 'INTEGER', primary_key=True),
        PostgresColumn('name', 'TEXT NOT NULL'),
        PostgresColumn('email', 'TEXT'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('country_code', 'CHAR(2)'),
        PostgresColumn('is_active', 'BOOLEAN NOT NULL'),
        PostgresColumn('profile', 'JSONB NOT NULL'),
    ]

    customer = [
        PostgresColumn('customer_id', 'INTEGER', primary_key=True),
        PostgresColumn('person_id', 'INTEGER NOT NULL', references='person(person_id)'),
        PostgresColumn('snapshot_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('name', 'TEXT NOT NULL'),
        PostgresColumn('email', 'TEXT'),
        PostgresColumn('country_code', 'CHAR(2)'),
        PostgresColumn('is_active', 'BOOLEAN NOT NULL'),
        PostgresColumn('profile', 'JSONB NOT NULL'),
    ]

    seller = [
        PostgresColumn('seller_id', 'INTEGER', primary_key=True),
        PostgresColumn('display_name', 'TEXT NOT NULL'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('country_code', 'CHAR(2)'),
        PostgresColumn('is_active', 'BOOLEAN NOT NULL'),
    ]

    product = [
        PostgresColumn('product_id', 'INTEGER', primary_key=True),
        PostgresColumn('seller_id', 'INTEGER NOT NULL', references='seller(seller_id)'),
        PostgresColumn('sku', 'TEXT'),
        PostgresColumn('title', 'TEXT NOT NULL'),
        PostgresColumn('description', 'TEXT'),
        PostgresColumn('price_cents', 'INTEGER NOT NULL'),
        # CHECK (price_cents >= 0)
        PostgresColumn('currency', 'CHAR(3) NOT NULL'),
        PostgresColumn('stock_qty', 'INTEGER NOT NULL'),
        # CHECK (stock_qty >= 0)
        PostgresColumn('is_active', 'BOOLEAN NOT NULL'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('updated_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('attributes', 'JSONB NOT NULL'),
    ]

    order = [
        PostgresColumn('order_id', 'INTEGER', primary_key=True),
        PostgresColumn('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
        PostgresColumn('ordered_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('status', 'TEXT NOT NULL'),
        # CHECK (status IN ('created', 'paid', 'shipped', 'canceled', 'refunded'))
        PostgresColumn('total_cents', 'INTEGER NOT NULL'),
        # CHECK (total_cents >= 0)
        PostgresColumn('currency', 'CHAR(3) NOT NULL'),
        PostgresColumn('shipping', 'JSONB NOT NULL'),
        PostgresColumn('payment', 'JSONB NOT NULL'),
    ]

    order_item = [
        PostgresColumn('order_item_id', 'INTEGER', primary_key=True),
        PostgresColumn('order_id', 'INTEGER NOT NULL', references='order(order_id)'),
        # ON DELETE CASCADE
        PostgresColumn('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
        PostgresColumn('unit_price_cents', 'INTEGER NOT NULL'),
        # CHECK (unit_price_cents >= 0)
        PostgresColumn('quantity', 'INTEGER NOT NULL'),
        # CHECK (quantity > 0)
        PostgresColumn('line_total_cents', 'INTEGER'),
        # GENERATED ALWAYS AS (unit_price_cents * quantity) STORED
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    review = [
        PostgresColumn('review_id', 'INTEGER', primary_key=True),
        PostgresColumn('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
        # ON DELETE CASCADE
        PostgresColumn('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
        # ON DELETE CASCADE
        PostgresColumn('rating', 'SMALLINT NOT NULL'),
        # CHECK (rating BETWEEN 1 AND 5)
        PostgresColumn('title', 'TEXT'),
        PostgresColumn('body', 'TEXT'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('helpful_votes', 'INTEGER NOT NULL'),
        # CHECK (helpful_votes >= 0)
    ]

    category = [
        PostgresColumn('category_id', 'INTEGER', primary_key=True),
        PostgresColumn('name', 'TEXT NOT NULL'),
        # Precomputed path helps both SQL and export to graph
        PostgresColumn('path', 'TEXT NOT NULL'),
    ]

    # Product has category (many to many)
    has_category = [
        PostgresColumn('product_id', 'INTEGER NOT NULL', primary_key=True, references='product(product_id)'),
        # ON DELETE CASCADE
        PostgresColumn('category_id', 'INTEGER NOT NULL', primary_key=True, references='category(category_id)'),
        # ON DELETE CASCADE
        PostgresColumn('assigned_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    # Person likes category (many to many)
    has_interest = [
        PostgresColumn('person_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
        # ON DELETE CASCADE
        PostgresColumn('category_id', 'INTEGER NOT NULL', primary_key=True, references='category(category_id)'),
        # ON DELETE CASCADE
        PostgresColumn('strength', 'SMALLINT NOT NULL'),
        # CHECK (strength BETWEEN 1 AND 10)
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    # Follows is person-person (graphy)
    follows = [
        PostgresColumn('from_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
        # ON DELETE CASCADE
        PostgresColumn('to_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
        # ON DELETE CASCADE
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        # CHECK (from_id <> to_id)
    ]

    return {
        'person': person,
        'customer': customer,
        'seller': seller,
        'product': product,
        'order': order,
        'order_item': order_item,
        'review': review,
        'category': category,
        'has_category': has_category,
        'has_interest': has_interest,
        'follows': follows,
    }
