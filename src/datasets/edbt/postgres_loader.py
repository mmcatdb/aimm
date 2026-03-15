from typing_extensions import override
from common.loaders.postgres_loader import PostgresLoader, ColumnSchema, IndexSchema

# TODO review is for customer_id, not person_id
# TODO order_item should not have seller_id (can get from product)

class EdbtPostgresLoader(PostgresLoader):
    @override
    def name(self):
        return 'EDBT'

    @override
    def _get_schemas(self):
        person = [
            ColumnSchema('person_id', 'INTEGER', primary_key=True),
            ColumnSchema('name', 'TEXT NOT NULL'),
            ColumnSchema('email', 'TEXT'),
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('country_code', 'CHAR(2)'),
            ColumnSchema('is_active', 'BOOLEAN NOT NULL'),
            ColumnSchema('profile', 'JSONB NOT NULL'),
        ]

        customer = [
            ColumnSchema('customer_id', 'INTEGER', primary_key=True),
            ColumnSchema('person_id', 'INTEGER NOT NULL', references='person(person_id)'),
            ColumnSchema('snapshot_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('name', 'TEXT NOT NULL'),
            ColumnSchema('email', 'TEXT'),
            ColumnSchema('country_code', 'CHAR(2)'),
            ColumnSchema('is_active', 'BOOLEAN NOT NULL'),
            ColumnSchema('profile', 'JSONB NOT NULL'),
        ]

        seller = [
            ColumnSchema('seller_id', 'INTEGER', primary_key=True),
            ColumnSchema('display_name', 'TEXT NOT NULL'),
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('country_code', 'CHAR(2)'),
            ColumnSchema('is_active', 'BOOLEAN NOT NULL'),
        ]

        product = [
            ColumnSchema('product_id', 'INTEGER', primary_key=True),
            ColumnSchema('seller_id', 'INTEGER NOT NULL', references='seller(seller_id)'),
            ColumnSchema('sku', 'TEXT'),
            ColumnSchema('title', 'TEXT NOT NULL'),
            ColumnSchema('description', 'TEXT'),
            ColumnSchema('price_cents', 'INTEGER NOT NULL'),
            # CHECK (price_cents >= 0)
            ColumnSchema('currency', 'CHAR(3) NOT NULL'),
            ColumnSchema('stock_qty', 'INTEGER NOT NULL'),
            # CHECK (stock_qty >= 0)
            ColumnSchema('is_active', 'BOOLEAN NOT NULL'),
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('updated_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('attributes', 'JSONB NOT NULL'),
        ]

        order = [
            ColumnSchema('order_id', 'INTEGER', primary_key=True),
            ColumnSchema('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
            ColumnSchema('ordered_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('status', 'TEXT NOT NULL'),
            # CHECK (status IN ('created', 'paid', 'shipped', 'canceled', 'refunded'))
            ColumnSchema('total_cents', 'INTEGER NOT NULL'),
            # CHECK (total_cents >= 0)
            ColumnSchema('currency', 'CHAR(3) NOT NULL'),
            ColumnSchema('shipping', 'JSONB NOT NULL'),
            ColumnSchema('payment', 'JSONB NOT NULL'),
        ]

        order_item = [
            ColumnSchema('order_item_id', 'INTEGER', primary_key=True),
            ColumnSchema('order_id', 'INTEGER NOT NULL', references='order(order_id)'),
            # ON DELETE CASCADE
            ColumnSchema('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
            ColumnSchema('unit_price_cents', 'INTEGER NOT NULL'),
            # CHECK (unit_price_cents >= 0)
            ColumnSchema('quantity', 'INTEGER NOT NULL'),
            # CHECK (quantity > 0)
            ColumnSchema('line_total_cents', 'INTEGER'),
            # GENERATED ALWAYS AS (unit_price_cents * quantity) STORED
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
            # Optional "doc style" snapshot for fast reads if you later embed in Mongo
            ColumnSchema('product_snapshot', 'JSONB NOT NULL'),
        ]

        review = [
            ColumnSchema('review_id', 'INTEGER', primary_key=True),
            ColumnSchema('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
            # ON DELETE CASCADE
            ColumnSchema('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
            # ON DELETE CASCADE
            ColumnSchema('rating', 'SMALLINT NOT NULL'),
            # CHECK (rating BETWEEN 1 AND 5)
            ColumnSchema('title', 'TEXT'),
            ColumnSchema('body', 'TEXT'),
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
            ColumnSchema('helpful_votes', 'INTEGER NOT NULL'),
            # CHECK (helpful_votes >= 0)
        ]

        category = [
            ColumnSchema('category_id', 'INTEGER', primary_key=True),
            ColumnSchema('name', 'TEXT NOT NULL'),
            # Precomputed path helps both SQL and export to graph
            ColumnSchema('path', 'TEXT NOT NULL'),
        ]

        # Product has category (many to many)
        has_category = [
            ColumnSchema('product_id', 'INTEGER NOT NULL', primary_key=True, references='product(product_id)'),
            # ON DELETE CASCADE
            ColumnSchema('category_id', 'INTEGER NOT NULL', primary_key=True, references='category(category_id)'),
            # ON DELETE CASCADE
            ColumnSchema('assigned_at', 'TIMESTAMPTZ NOT NULL'),
        ]

        # Person likes category (many to many)
        has_interest = [
            ColumnSchema('person_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
            # ON DELETE CASCADE
            ColumnSchema('category_id', 'INTEGER NOT NULL', primary_key=True, references='category(category_id)'),
            # ON DELETE CASCADE
            ColumnSchema('strength', 'SMALLINT NOT NULL'),
            # CHECK (strength BETWEEN 1 AND 10)
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
        ]

        # Follows is person-person (graphy)
        follows = [
            ColumnSchema('from_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
            # ON DELETE CASCADE
            ColumnSchema('to_id', 'INTEGER NOT NULL', primary_key=True, references='person(person_id)'),
            # ON DELETE CASCADE
            ColumnSchema('created_at', 'TIMESTAMPTZ NOT NULL'),
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

    @override
    def _get_indexes(self):
        return [
            IndexSchema('person', [ 'email' ], is_unique=True),
            IndexSchema('product', [ 'sku' ], is_unique=True),
            IndexSchema('product', [ 'seller_id' ]),
            IndexSchema('product', [ 'is_active' ], where='"is_active" = TRUE'),
            IndexSchema('order', [ 'customer_id', 'ordered_at' ]),
            # ordered_at DESC
            IndexSchema('order', [ 'ordered_at' ]),
            # ordered_at DESC
            IndexSchema('order_item', [ 'order_id' ]),
            IndexSchema('order_item', [ 'product_id' ]),
            IndexSchema('review', [ 'product_id', 'customer_id' ], is_unique=True),
            IndexSchema('review', [ 'product_id', 'created_at' ]),
            # created_at DESC
            IndexSchema('review', [ 'customer_id', 'created_at' ]),
            # created_at DESC
            IndexSchema('category', [ 'path' ], is_unique=True),
            IndexSchema('has_category', [ 'category_id' ]),
            IndexSchema('has_interest', [ 'category_id' ]),
            IndexSchema('follows', [ 'to_id' ]),
        ]
