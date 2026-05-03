from typing_extensions import override

from core.loaders.postgres_loader import PostgresColumn, PostgresIndex, PostgresLoader


class EdbtPostgresLoader(PostgresLoader):
    @override
    def _get_kinds(self) -> dict[str, list[PostgresColumn]]:
        c = PostgresColumn
        return {
            'person': [
                c('person_id', 'INTEGER', primary_key=True),
                c('full_name', 'TEXT NOT NULL'),
                c('email', 'TEXT'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
                c('country_code', 'CHAR(2)'),
                c('is_active', 'BOOLEAN NOT NULL'),
                c('profile', 'JSONB NOT NULL'),
            ],
            'customer': [
                c('customer_id', 'INTEGER', primary_key=True),
                c('person_id', 'INTEGER NOT NULL', references='person(person_id)'),
                c('snapshot_at', 'TIMESTAMPTZ NOT NULL'),
                c('full_name', 'TEXT NOT NULL'),
                c('email', 'TEXT'),
                c('country_code', 'CHAR(2)'),
                c('is_active', 'BOOLEAN NOT NULL'),
                c('profile', 'JSONB NOT NULL'),
            ],
            'seller': [
                c('seller_id', 'INTEGER', primary_key=True),
                c('display_name', 'TEXT NOT NULL'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
                c('country_code', 'CHAR(2)'),
                c('is_active', 'BOOLEAN NOT NULL'),
            ],
            'product': [
                c('product_id', 'INTEGER', primary_key=True),
                c('seller_id', 'INTEGER NOT NULL', references='seller(seller_id)'),
                c('sku', 'TEXT NOT NULL'),
                c('title', 'TEXT NOT NULL'),
                c('description', 'TEXT'),
                c('price_cents', 'INTEGER NOT NULL'),
                c('currency', 'CHAR(3) NOT NULL'),
                c('stock_qty', 'INTEGER NOT NULL'),
                c('is_active', 'BOOLEAN NOT NULL'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
                c('updated_at', 'TIMESTAMPTZ NOT NULL'),
                c('attributes', 'JSONB NOT NULL'),
            ],
            'category': [
                c('category_id', 'INTEGER', primary_key=True),
                c('name', 'TEXT NOT NULL'),
                c('path', 'TEXT NOT NULL'),
            ],
            'has_category': [
                c('product_id', 'INTEGER', primary_key=True, references='product(product_id)'),
                c('category_id', 'INTEGER', primary_key=True, references='category(category_id)'),
                c('assigned_at', 'TIMESTAMPTZ NOT NULL'),
            ],
            'has_interest': [
                c('person_id', 'INTEGER', primary_key=True, references='person(person_id)'),
                c('category_id', 'INTEGER', primary_key=True, references='category(category_id)'),
                c('strength', 'SMALLINT NOT NULL'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
            ],
            'follows': [
                c('from_id', 'INTEGER', primary_key=True, references='person(person_id)'),
                c('to_id', 'INTEGER', primary_key=True, references='person(person_id)'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
            ],
            'order': [
                c('order_id', 'INTEGER', primary_key=True),
                c('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
                c('ordered_at', 'TIMESTAMPTZ NOT NULL'),
                c('status', 'TEXT NOT NULL'),
                c('total_cents', 'INTEGER NOT NULL'),
                c('currency', 'CHAR(3) NOT NULL'),
                c('shipping', 'JSONB NOT NULL'),
                c('payment', 'JSONB NOT NULL'),
            ],
            'order_item': [
                c('order_item_id', 'INTEGER', primary_key=True),
                c('order_id', 'INTEGER NOT NULL', references='order(order_id)'),
                c('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
                c('unit_price_cents', 'INTEGER NOT NULL'),
                c('quantity', 'INTEGER NOT NULL'),
                c('line_total_cents', 'INTEGER NOT NULL'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
                c('product_snapshot', 'JSONB NOT NULL'),
            ],
            'review': [
                c('review_id', 'INTEGER', primary_key=True),
                c('product_id', 'INTEGER NOT NULL', references='product(product_id)'),
                c('customer_id', 'INTEGER NOT NULL', references='customer(customer_id)'),
                c('rating', 'SMALLINT NOT NULL'),
                c('title', 'TEXT'),
                c('body', 'TEXT'),
                c('created_at', 'TIMESTAMPTZ NOT NULL'),
                c('helpful_votes', 'INTEGER NOT NULL'),
            ],
        }

    @override
    def _get_constraints(self) -> list[PostgresIndex]:
        return [
            PostgresIndex('person', ['email'], is_unique=True),
            PostgresIndex('customer', ['person_id']),
            PostgresIndex('seller', ['country_code']),
            PostgresIndex('product', ['seller_id']),
            PostgresIndex('product', ['is_active'], where='"is_active" = TRUE'),
            PostgresIndex('order', ['customer_id']),
            PostgresIndex('order', ['ordered_at']),
            PostgresIndex('order_item', ['order_id']),
            PostgresIndex('order_item', ['product_id']),
            PostgresIndex('review', ['product_id', 'customer_id'], is_unique=True),
            PostgresIndex('has_category', ['category_id']),
            PostgresIndex('has_interest', ['category_id']),
            PostgresIndex('follows', ['to_id']),
        ]


def export():
    return EdbtPostgresLoader()
