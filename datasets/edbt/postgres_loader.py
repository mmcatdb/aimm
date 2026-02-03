from typing_extensions import override
from common.config import Config
from common.drivers import PostgresDriver
from common.loaders.postgres_loader import PostgresLoader

class EdbtPostgresLoader(PostgresLoader):
    def __init__(self, config: Config, driver: PostgresDriver):
        super().__init__(config, driver)

    @override
    def name(self) -> str:
        return 'EDBT'

    @override
    def _get_schemas(self) -> dict[str, list[dict]]:
        user = [
            { 'name': 'user_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'handle', 'type': 'TEXT NOT NULL' },
            { 'name': 'email', 'type': 'TEXT' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'country_code', 'type': 'CHAR(2)' },
            { 'name': 'is_active', 'type': 'BOOLEAN NOT NULL' },
            { 'name': 'profile', 'type': 'JSONB NOT NULL' },
        ]

        seller = [
            { 'name': 'seller_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'display_name', 'type': 'TEXT NOT NULL' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'country_code', 'type': 'CHAR(2)' },
            { 'name': 'is_active', 'type': 'BOOLEAN NOT NULL' },
        ]

        product = [
            { 'name': 'product_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'seller_id', 'type': 'INTEGER NOT NULL', 'references': 'seller(seller_id)' },
            { 'name': 'sku', 'type': 'TEXT' },
            { 'name': 'title', 'type': 'TEXT NOT NULL' },
            { 'name': 'description', 'type': 'TEXT' },
            { 'name': 'price_cents', 'type': 'INTEGER NOT NULL' },
            # CHECK (price_cents >= 0)
            { 'name': 'currency', 'type': 'CHAR(3) NOT NULL' },
            { 'name': 'stock_qty', 'type': 'INTEGER NOT NULL' },
            # CHECK (stock_qty >= 0)
            { 'name': 'is_active', 'type': 'BOOLEAN NOT NULL' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'updated_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'attributes', 'type': 'JSONB NOT NULL' },
        ]

        order = [
            { 'name': 'order_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'buyer_user_id', 'type': 'INTEGER NOT NULL', 'references': 'user(user_id)' },
            { 'name': 'order_ts', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'status', 'type': 'TEXT NOT NULL' },
            # CHECK (status IN ('created','paid','shipped','cancelled','refunded'))
            { 'name': 'total_cents', 'type': 'INTEGER NOT NULL' },
            # CHECK (total_cents >= 0)
            { 'name': 'currency', 'type': 'CHAR(3) NOT NULL' },
            { 'name': 'shipping', 'type': 'JSONB NOT NULL' },
            { 'name': 'payment', 'type': 'JSONB NOT NULL' },
        ]

        order_item = [
            { 'name': 'order_item_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'order_id', 'type': 'INTEGER NOT NULL', 'references': 'order(order_id)' },
            # ON DELETE CASCADE
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'references': 'product(product_id)' },
            { 'name': 'seller_id', 'type': 'INTEGER NOT NULL', 'references': 'seller(seller_id)' },
            { 'name': 'unit_price_cents', 'type': 'INTEGER NOT NULL' },
            # CHECK (unit_price_cents >= 0)
            { 'name': 'quantity', 'type': 'INTEGER NOT NULL' },
            # CHECK (quantity > 0)
            { 'name': 'line_total_cents', 'type': 'INTEGER' },
            # GENERATED ALWAYS AS (unit_price_cents * quantity) STORED
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            # Optional “doc style” snapshot for fast reads if you later embed in Mongo
            { 'name': 'product_snapshot', 'type': 'JSONB NOT NULL' },
        ]

        review = [
            { 'name': 'review_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'references': 'product(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'user_id', 'type': 'INTEGER NOT NULL', 'references': 'user(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'rating', 'type': 'SMALLINT NOT NULL' },
            # CHECK (rating BETWEEN 1 AND 5)
            { 'name': 'title', 'type': 'TEXT' },
            { 'name': 'body', 'type': 'TEXT' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'helpful_votes', 'type': 'INTEGER NOT NULL' },
            # CHECK (helpful_votes >= 0)
        ]

        category = [
            { 'name': 'category_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'name', 'type': 'TEXT NOT NULL' },
            # Precomputed path helps both SQL and export to graph
            { 'name': 'path', 'type': 'TEXT NOT NULL' },
        ]

        # Product has category (many to many)
        has_category = [
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'product(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'category_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'category(category_id)' },
            # ON DELETE CASCADE
            { 'name': 'assigned_at', 'type': 'TIMESTAMPTZ NOT NULL' },
        ]

        # User likes category (many to many)
        has_interest = [
            { 'name': 'user_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'user(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'category_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'category(category_id)' },
            # ON DELETE CASCADE
            { 'name': 'strength', 'type': 'SMALLINT NOT NULL' },
            # CHECK (strength BETWEEN 1 AND 10)
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
        ]

        # Follows is user-user (graphy)
        follows = [
            { 'name': 'from_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'user(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'to_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'user(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            # CHECK (from_id <> to_id)
        ]

        return {
            'user': user,
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
    def _get_indexes(self) -> list[dict]:
        return [
            { 'table': 'user', 'columns': [ 'handle' ], 'unique': True },
            { 'table': 'user', 'columns': [ 'email' ], 'unique': True },
            { 'table': 'product', 'columns': [ 'sku' ], 'unique': True },
            { 'table': 'product', 'columns': [ 'seller_id' ], },
            { 'table': 'product', 'columns': [ 'is_active' ], 'where': '"is_active" = TRUE' },
            { 'table': 'order', 'columns': [ 'buyer_user_id', 'order_ts' ], },
            # order_ts DESC
            { 'table': 'order', 'columns': [ 'order_ts' ], },
            # order_ts DESC
            { 'table': 'order_item', 'columns': [ 'order_id' ], },
            { 'table': 'order_item', 'columns': [ 'product_id' ], },
            { 'table': 'order_item', 'columns': [ 'seller_id' ], },
            { 'table': 'review', 'columns': [ 'product_id', 'user_id' ], 'unique': True },
            { 'table': 'review', 'columns': [ 'product_id', 'created_at' ], },
            # created_at DESC
            { 'table': 'review', 'columns': [ 'user_id', 'created_at' ], },
            # created_at DESC
            { 'table': 'category', 'columns': [ 'path' ], 'unique': True },
            { 'table': 'has_category', 'columns': [ 'category_id' ], },
            { 'table': 'has_interest', 'columns': [ 'category_id' ], },
            { 'table': 'follows', 'columns': [ 'to_id' ], },
        ]
