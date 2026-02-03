from typing_extensions import override
from common.config import Config
from common.drivers import PostgresDriver
from common.loaders.postgres_loader import PostgresLoader

# TODO
# Rename follower and followee to from and to
# Remove parent_category_id from categories

class EdbtPostgresLoader(PostgresLoader):
    def __init__(self, config: Config, driver: PostgresDriver):
        super().__init__(config, driver)

    @override
    def name(self) -> str:
        return 'EDBT'

    @override
    def _get_schemas(self) -> dict[str, list[dict]]:
        users = [
            { 'name': 'user_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'handle', 'type': 'TEXT NOT NULL' },
            { 'name': 'email', 'type': 'TEXT' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'country_code', 'type': 'CHAR(2)' },
            { 'name': 'is_active', 'type': 'BOOLEAN NOT NULL' },
            { 'name': 'profile', 'type': 'JSONB NOT NULL' },
        ]

        sellers = [
            { 'name': 'seller_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'display_name', 'type': 'TEXT NOT NULL' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'country_code', 'type': 'CHAR(2)' },
            { 'name': 'is_active', 'type': 'BOOLEAN NOT NULL' },
        ]

        products = [
            { 'name': 'product_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'seller_id', 'type': 'INTEGER NOT NULL', 'references': 'sellers(seller_id)' },
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

        orders = [
            { 'name': 'order_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'buyer_user_id', 'type': 'INTEGER NOT NULL', 'references': 'users(user_id)' },
            { 'name': 'order_ts', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'status', 'type': 'TEXT NOT NULL' },
            # CHECK (status IN ('created','paid','shipped','cancelled','refunded'))
            { 'name': 'total_cents', 'type': 'INTEGER NOT NULL' },
            # CHECK (total_cents >= 0)
            { 'name': 'currency', 'type': 'CHAR(3) NOT NULL' },
            { 'name': 'shipping', 'type': 'JSONB NOT NULL' },
            { 'name': 'payment', 'type': 'JSONB NOT NULL' },
        ]

        order_items = [
            { 'name': 'order_item_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'order_id', 'type': 'INTEGER NOT NULL', 'references': 'orders(order_id)' },
            # ON DELETE CASCADE
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'references': 'products(product_id)' },
            { 'name': 'seller_id', 'type': 'INTEGER NOT NULL', 'references': 'sellers(seller_id)' },
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

        reviews = [
            { 'name': 'review_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'references': 'products(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'user_id', 'type': 'INTEGER NOT NULL', 'references': 'users(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'rating', 'type': 'SMALLINT NOT NULL' },
            # CHECK (rating BETWEEN 1 AND 5)
            { 'name': 'title', 'type': 'TEXT' },
            { 'name': 'body', 'type': 'TEXT' },
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            { 'name': 'helpful_votes', 'type': 'INTEGER NOT NULL' },
            # CHECK (helpful_votes >= 0)
        ]

        categories = [
            { 'name': 'category_id', 'type': 'INTEGER', 'primary_key': True },
            { 'name': 'name', 'type': 'TEXT NOT NULL' },
            # Precomputed path helps both SQL and export to graph
            { 'name': 'path', 'type': 'TEXT NOT NULL' },
        ]

        # Product has category (many to many)
        has_category = [
            { 'name': 'product_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'products(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'category_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'categories(category_id)' },
            # ON DELETE CASCADE
            { 'name': 'assigned_at', 'type': 'TIMESTAMPTZ NOT NULL' },
        ]

        # User likes category (many to many)
        has_interest = [
            { 'name': 'user_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'users(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'category_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'categories(category_id)' },
            # ON DELETE CASCADE
            { 'name': 'strength', 'type': 'SMALLINT NOT NULL' },
            # CHECK (strength BETWEEN 1 AND 10)
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
        ]

        # Follows is user-user (graphy)
        follows = [
            { 'name': 'follower_user_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'users(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'followee_user_id', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'users(user_id)' },
            # ON DELETE CASCADE
            { 'name': 'created_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            # CHECK (follower_user_id <> followee_user_id)
        ]

        # Similar is product-product (symmetric)
        # Store each pair once, enforce product_id_a < product_id_b.
        similar = [
            { 'name': 'product_id_a', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'products(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'product_id_b', 'type': 'INTEGER NOT NULL', 'primary_key': True, 'references': 'products(product_id)' },
            # ON DELETE CASCADE
            { 'name': 'score', 'type': 'DOUBLE PRECISION NOT NULL' },
            # CHECK (score >= 0)
            { 'name': 'source', 'type': 'TEXT NOT NULL' },
            { 'name': 'updated_at', 'type': 'TIMESTAMPTZ NOT NULL' },
            # CHECK (product_id_a < product_id_b)
        ]

        return {
            'users': users,
            'sellers': sellers,
            'products': products,
            'orders': orders,
            'order_items': order_items,
            'reviews': reviews,
            'categories': categories,
            'has_category': has_category,
            'has_interest': has_interest,
            'follows': follows,
            'similar': similar,
        }

    @override
    def _get_indexes(self) -> list[dict]:
        return [
            { 'table': 'users', 'columns': [ 'handle' ], 'unique': True },
            { 'table': 'users', 'columns': [ 'email' ], 'unique': True },
            { 'table': 'products', 'columns': [ 'sku' ], 'unique': True },
            { 'table': 'products', 'columns': [ 'seller_id' ], },
            { 'table': 'products', 'columns': [ 'is_active' ], 'where': '"is_active" = TRUE' },
            { 'table': 'orders', 'columns': [ 'buyer_user_id', 'order_ts' ], },
            # order_ts DESC
            { 'table': 'orders', 'columns': [ 'order_ts' ], },
            # order_ts DESC
            { 'table': 'order_items', 'columns': [ 'order_id' ], },
            { 'table': 'order_items', 'columns': [ 'product_id' ], },
            { 'table': 'order_items', 'columns': [ 'seller_id' ], },
            { 'table': 'reviews', 'columns': [ 'product_id', 'user_id' ], 'unique': True },
            { 'table': 'reviews', 'columns': [ 'product_id', 'created_at' ], },
            # created_at DESC
            { 'table': 'reviews', 'columns': [ 'user_id', 'created_at' ], },
            # created_at DESC
            { 'table': 'categories', 'columns': [ 'path' ], 'unique': True },
            { 'table': 'has_category', 'columns': [ 'category_id' ], },
            { 'table': 'has_interest', 'columns': [ 'category_id' ], },
            { 'table': 'follows', 'columns': [ 'followee_user_id' ], },
            { 'table': 'similar', 'columns': [ 'product_id_a' ], },
            { 'table': 'similar', 'columns': [ 'product_id_b' ], },
        ]
