from typing_extensions import override
from core.loaders.mongo_loader import MongoLoader, MongoPostgresBuilder, MongoIndex
from ...postgres.edbt.loader import get_postgres_edbt_kinds

def export():
    return MongoEdbtLoader()

class MongoEdbtLoader(MongoLoader):

    @override
    def _get_kinds(self):
        b = MongoPostgresBuilder.create(get_postgres_edbt_kinds())

        person = b.document('person', {
            'person_id': 'person_id',
            'name': 'name',
            'email': 'email',
            'created_at': 'created_at',
            'country_code': 'country_code',
            'is_active': 'is_active',
            'profile': 'profile',

            'interests': b.nested('has_interest', {
                'strength': 'strength',
                'created_at': 'created_at',

                'category': b.nested('category', {
                    'category_id': 'category_id',
                    'name': 'name',
                    'path': 'path',
                }, parent_join='category_id', child_join='category_id'),

            }, parent_join='person_id', child_join='person_id', is_array=True),

            'follows': b.nested('follows', {
                'created_at': 'created_at',

                'to_person': b.nested('person', {
                    'person_id': 'person_id',
                    'name': 'name',
                    'country_code': 'country_code',
                    'is_active': 'is_active',
                }, parent_join='to_id', child_join='person_id'),

            }, parent_join='person_id', child_join='from_id', is_array=True),

        })

        product = b.document('product', {
            'product_id': 'product_id',
            'sku': 'sku',
            'title': 'title',
            'description': 'description',
            'price_cents': 'price_cents',
            'currency': 'currency',
            'stock_qty': 'stock_qty',
            'is_active': 'is_active',
            'created_at': 'created_at',
            'updated_at': 'updated_at',
            'attributes': 'attributes',

            'seller': b.nested('seller', {
                'seller_id': 'seller_id',
                'display_name': 'display_name',
                'country_code': 'country_code',
                'is_active': 'is_active',
            }, parent_join='seller_id', child_join='seller_id'),

            'categories': b.nested('has_category', {
                'assigned_at': 'assigned_at',

                'category': b.nested('category', {
                    'category_id': 'category_id',
                    'name': 'name',
                    'path': 'path',
                }, parent_join='category_id', child_join='category_id'),

            }, parent_join='product_id', child_join='product_id', is_array=True),

            'reviews': b.nested('review', {
                'review_id': 'review_id',
                'customer_id': 'customer_id',
                'rating': 'rating',
                'title': 'title',
                'body': 'body',
                'created_at': 'created_at',
                'helpful_votes': 'helpful_votes',
            }, parent_join='product_id', child_join='product_id', is_array=True),

        })

        order = b.document('order', {
            'order_id': 'order_id',
            'ordered_at': 'ordered_at',
            'status': 'status',
            'total_cents': 'total_cents',
            'currency': 'currency',
            'shipping': 'shipping',
            'payment': 'payment',

            'customer': b.nested('customer', {
                'customer_id': 'customer_id',
                'person_id': 'person_id',
                'snapshot_at': 'snapshot_at',
                'name': 'name',
                'email': 'email',
                'country_code': 'country_code',
                'is_active': 'is_active',
                'profile': 'profile',
            }, parent_join='customer_id', child_join='customer_id'),

            'items': b.nested('order_item', {
                'order_item_id': 'order_item_id',
                'unit_price_cents': 'unit_price_cents',
                'quantity': 'quantity',
                'line_total_cents': 'line_total_cents',
                'created_at': 'created_at',

                'product': b.nested('product', {
                    'product_id': 'product_id',
                    'seller_id': 'seller_id',
                    'sku': 'sku',
                    'title': 'title',
                    'description': 'description',
                    'price_cents': 'price_cents',
                    'currency': 'currency',
                    'created_at': 'created_at',
                    'updated_at': 'updated_at',
                    'attributes': 'attributes',
                }, parent_join='product_id', child_join='product_id'),

            }, parent_join='order_id', child_join='order_id', is_array=True),

        })

        return [
            person,
            b.plain_copy('customer'),
            b.plain_copy('seller'),
            b.plain_copy('category'),
            product,
            b.plain_copy('review'),
            order,
        ]

    @override
    def _get_constraints(self):
        return [
            MongoIndex('person', ['person_id'], is_unique=True),
            MongoIndex('person', ['country_code']),
            MongoIndex('person', ['is_active']),
            MongoIndex('person', ['interests.category.category_id']),
            MongoIndex('person', ['follows.to_person.person_id']),

            MongoIndex('customer', ['customer_id'], is_unique=True),
            MongoIndex('customer', ['person_id']),

            MongoIndex('seller', ['seller_id'], is_unique=True),

            MongoIndex('category', ['category_id'], is_unique=True),
            MongoIndex('category', ['path'], is_unique=True),

            MongoIndex('product', ['product_id'], is_unique=True),
            MongoIndex('product', ['is_active']),
            MongoIndex('product', ['seller.seller_id']),
            MongoIndex('product', ['categories.category.category_id']),
            MongoIndex('product', ['categories.category.path']),
            MongoIndex('product', ['reviews.total_count']),
            MongoIndex('product', ['reviews.average_rating']),

            MongoIndex('order', ['order_id'], is_unique=True),
            MongoIndex('order', ['ordered_at']),
            MongoIndex('order', ['status']),
            MongoIndex('order', ['customer.customer_id']),
            MongoIndex('order', ['customer.person_id']),
            MongoIndex('order', ['items.product_id']),

            MongoIndex('review', ['review_id'], is_unique=True),
            MongoIndex('review', ['product_id', 'customer_id'], is_unique=True),
            MongoIndex('review', ['product_id']),
            MongoIndex('review', ['customer_id']),
            MongoIndex('review', ['created_at']),
        ]
