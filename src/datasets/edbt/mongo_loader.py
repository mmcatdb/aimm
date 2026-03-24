from typing_extensions import override
from common.loaders.mongo_loader import MongoLoader, MongoPostgresBuilder, IndexSchema
from datasets.edbt.postgres_loader import get_postgres_edbt_schemas

class EdbtMongoLoader(MongoLoader):
    @override
    def name(self):
        return 'EDBT'

    @override
    def run(self, import_directory: str, do_reset: bool):
        super().run(import_directory, do_reset)
        self._build_review_summaries()

    @override
    def _get_schemas(self):
        b = MongoPostgresBuilder.create(get_postgres_edbt_schemas())

        person = b.document('person', {
            'person_id': 'person_id',
            'name': 'name',
            'email': 'email',
            'created_at': 'created_at',
            'country_code': 'country_code',
            'is_active': 'is_active',
            'profile': 'profile',
            'interests': b.nested('has_interest', {
                'category_id': 'category_id',
                'strength': 'strength',
                'created_at': 'created_at',
                'category': b.nested('category', {
                    'category_id': 'category_id',
                    'name': 'name',
                    'path': 'path',
                }, parent_join='category_id', child_join='category_id'),
            }, parent_join='person_id', child_join='person_id', is_array=True),
            'follows': b.nested('follows', {
                'to_id': 'to_id',
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
            'seller_id': 'seller_id',
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
                'category_id': 'category_id',
                'assigned_at': 'assigned_at',
                'category': b.nested('category', {
                    'category_id': 'category_id',
                    'name': 'name',
                    'path': 'path',
                }, parent_join='category_id', child_join='category_id'),
            }, parent_join='product_id', child_join='product_id', is_array=True),
        })

        order = b.document('order', {
            'order_id': 'order_id',
            'customer_id': 'customer_id',
            'ordered_at': 'ordered_at',
            'status': 'status',
            'total_cents': 'total_cents',
            'currency': 'currency',
            'shipping': 'shipping',
            'payment': 'payment',
            'customer': b.nested('customer', {
                'customer_id': 'customer_id',
                'name': 'name',
                'email': 'email',
                'country_code': 'country_code',
                'is_active': 'is_active',
                'profile': 'profile',
                'snapshot_at': 'snapshot_at',
            }, parent_join='customer_id', child_join='customer_id'),
            'items': b.nested('order_item', {
                'order_item_id': 'order_item_id',
                'product_id': 'product_id',
                'unit_price_cents': 'unit_price_cents',
                'quantity': 'quantity',
                'line_total_cents': 'line_total_cents',
                'created_at': 'created_at',
                'product_snapshot': 'product_snapshot',
                'product': b.nested('product', {
                    'product_id': 'product_id',
                    'sku': 'sku',
                    'title': 'title',
                    'currency': 'currency',
                }, parent_join='product_id', child_join='product_id'),
            }, parent_join='order_id', child_join='order_id', is_array=True),
        })

        return [
            person,
            b.plain_copy('customer'),
            b.plain_copy('seller'),
            b.plain_copy('category'),
            product,
            order,
            b.plain_copy('review'),
        ]

    @override
    def _get_indexes(self):
        return [
            IndexSchema('person', ['person_id'], is_unique=True),
            IndexSchema('person', ['country_code']),
            IndexSchema('person', ['is_active']),
            IndexSchema('person', ['interests.category.category_id']),
            IndexSchema('person', ['follows.to_id']),

            IndexSchema('customer', ['customer_id'], is_unique=True),
            IndexSchema('customer', ['person_id']),

            IndexSchema('seller', ['seller_id'], is_unique=True),

            IndexSchema('category', ['category_id'], is_unique=True),
            IndexSchema('category', ['path'], is_unique=True),

            IndexSchema('product', ['product_id'], is_unique=True),
            IndexSchema('product', ['seller_id']),
            IndexSchema('product', ['is_active']),
            IndexSchema('product', ['seller.seller_id']),
            IndexSchema('product', ['categories.category_id']),
            IndexSchema('product', ['categories.category.path']),
            IndexSchema('product', ['review_summary.total_count']),
            IndexSchema('product', ['review_summary.average_rating']),

            IndexSchema('order', ['order_id'], is_unique=True),
            IndexSchema('order', ['customer_id']),
            IndexSchema('order', ['ordered_at']),
            IndexSchema('order', ['status']),
            IndexSchema('order', ['customer.customer_id']),
            IndexSchema('order', ['items.product_id']),

            IndexSchema('review', ['review_id'], is_unique=True),
            IndexSchema('review', ['product_id', 'customer_id'], is_unique=True),
            IndexSchema('review', ['product_id']),
            IndexSchema('review', ['customer_id']),
            IndexSchema('review', ['created_at']),
        ]

    def _build_review_summaries(self):
        review_collection = self._driver.collection('review')
        product_collection = self._driver.collection('product')

        product_collection.update_many({}, {
            '$set': {
                'review_summary': {
                    'average_rating': None,
                    'total_count': 0,
                    'total_helpful_votes': 0,
                    'last_review_at': None,
                }
            }
        })

        summary_cursor = review_collection.aggregate([
            {
                '$group': {
                    '_id': '$product_id',
                    'average_rating': {'$avg': '$rating'},
                    'total_count': {'$sum': 1},
                    'total_helpful_votes': {'$sum': '$helpful_votes'},
                    'last_review_at': {'$max': '$created_at'},
                }
            }
        ])

        for summary in summary_cursor:
            product_collection.update_one(
                {'product_id': summary['_id']},
                {
                    '$set': {
                        'review_summary': {
                            'average_rating': float(summary['average_rating']),
                            'total_count': summary['total_count'],
                            'total_helpful_votes': summary['total_helpful_votes'],
                            'last_review_at': summary['last_review_at'],
                        }
                    }
                }
            )
