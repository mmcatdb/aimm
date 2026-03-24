from datetime import datetime, timedelta

from common.query_registry import query
from common.drivers import DriverType
from common.database import MongoQuery, MongoFindQuery, MongoAggregateQuery
from datasets.edbt.edbt_database import EdbtDatabase


class EdbtMongoDatabase(EdbtDatabase[MongoQuery]):

    def __init__(self):
        super().__init__(DriverType.MONGO)

    @query('test', 1.0, 'Product page read (embedded seller, categories, review summary)')
    def _product_page_read(self):
        return MongoFindQuery('product',
            filter={
                'product_id': self._param_product_id(),
                'is_active': True,
            },
            projection={
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'description': 1,
                'price_cents': 1,
                'currency': 1,
                'stock_qty': 1,
                'seller': 1,
                'review_summary': 1,
                'categories': {'$slice': 10},
            },
            limit=1,
        )

    @query('test', 1.0, 'Bulk fetch product pages for a feed (embedded-only)')
    def _bulk_fetch_product_pages(self):
        return MongoFindQuery('product',
            filter={
                'product_id': {'$in': self._param_product_ids(5, 20)},
                'is_active': True,
            },
            projection={
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'price_cents': 1,
                'currency': 1,
                'seller': 1,
                'review_summary': 1,
            },
            limit=200,
        )

    @query('test', 1.0, 'Top products by quality in one category')
    def _top_products_by_revenue(self):
        return MongoFindQuery('product',
            filter={
                'categories.category_id': self._param_category_id(),
                'is_active': True,
            },
            projection={
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'price_cents': 1,
                'currency': 1,
                'review_summary.average_rating': 1,
                'review_summary.total_count': 1,
            },
            sort={
                'review_summary.average_rating': -1,
                'review_summary.total_count': -1,
                'price_cents': -1,
            },
            limit=50,
        )

    @query('test', 1.0, 'Seller catalog snapshot (active products only)')
    def _seller_daily_revenue(self):
        return MongoFindQuery('product',
            filter={
                'seller_id': self._param_seller_id(),
                'is_active': True,
            },
            projection={
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'price_cents': 1,
                'currency': 1,
                'stock_qty': 1,
                'review_summary': 1,
            },
            sort={
                'review_summary.total_count': -1,
                'price_cents': -1,
            },
            limit=200,
        )

    @query('test', 1.0, 'Customer spend buckets over last 90 days (per customer snapshot)')
    def _customer_spend_buckets(self):
        start_date = self._param('start_date', lambda: datetime.now() - timedelta(days=90))

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': start_date},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$group': {
                '_id': '$customer.customer_id',
                'spend_cents': {'$sum': '$total_cents'},
            }},
            {'$project': {
                'bucket': {
                    '$switch': {
                        'branches': [
                            {'case': {'$lt': ['$spend_cents', 5000]}, 'then': 'low'},
                            {'case': {'$lt': ['$spend_cents', 20000]}, 'then': 'mid'},
                        ],
                        'default': 'high',
                    }
                }
            }},
            {'$group': {
                '_id': '$bucket',
                'customers': {'$sum': 1},
            }},
            {'$sort': {'_id': 1}},
        ])

    @query('test', 1.0, 'High-velocity customer pattern in last 24h (embedded order items)')
    def _fraud_pattern(self):
        start_date = self._param('start_date', lambda: datetime.now() - timedelta(hours=24))

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': start_date},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$project': {
                'customer_id': '$customer.customer_id',
                'order_id': '$order_id',
                'product_ids': {
                    '$map': {
                        'input': '$items',
                        'as': 'item',
                        'in': '$$item.product_id',
                    }
                },
            }},
            {'$group': {
                '_id': '$customer_id',
                'orders': {'$sum': 1},
                'product_sets': {'$push': '$product_ids'},
            }},
            {'$project': {
                'orders': 1,
                'distinct_products': {
                    '$size': {
                        '$reduce': {
                            'input': '$product_sets',
                            'initialValue': [],
                            'in': {'$setUnion': ['$$value', '$$this']},
                        }
                    }
                },
            }},
            {'$match': {
                'orders': {'$gte': 5},
                'distinct_products': {'$gte': 10},
            }},
            {'$sort': {'distinct_products': -1, 'orders': -1}},
            {'$limit': 200},
        ])

    @query('test', 1.0, 'Follow graph candidates from embedded follows')
    def _who_to_follow(self):
        person_id = self._param_person_id()

        return MongoAggregateQuery('person', [
            {'$match': {'follows.to_id': person_id}},
            {'$group': {
                '_id': '$person_id',
                'paths': {'$sum': 1},
            }},
            {'$sort': {'paths': -1}},
            {'$limit': 50},
        ])

    @query('test', 1.0, 'People also bought for a target product (same-order co-buy)')
    def _people_also_bought(self):
        product_id = self._param_product_id()

        return MongoAggregateQuery('order', [
            {'$match': {
                'status': {'$in': ['paid', 'shipped']},
                'items.product_id': product_id,
            }},
            {'$project': {
                'other_product_ids': {
                    '$filter': {
                        'input': {
                            '$map': {
                                'input': '$items',
                                'as': 'item',
                                'in': '$$item.product_id',
                            }
                        },
                        'as': 'pid',
                        'cond': {'$ne': ['$$pid', product_id]},
                    }
                }
            }},
            {'$unwind': '$other_product_ids'},
            {'$group': {
                '_id': '$other_product_ids',
                'co_buy': {'$sum': 1},
            }},
            {'$sort': {'co_buy': -1}},
            {'$limit': 20},
        ])

    @query('test', 1.0, 'Personalized feed category candidates from embedded person interests')
    def _personalized_feed_candidates(self):
        return MongoAggregateQuery('person', [
            {'$match': {
                'person_id': self._param_person_id(),
                'is_active': True,
            }},
            {'$unwind': '$interests'},
            {'$group': {
                '_id': '$interests.category_id',
                'interest_score': {'$sum': '$interests.strength'},
                'category_name': {'$max': '$interests.category.name'},
                'category_path': {'$max': '$interests.category.path'},
            }},
            {'$sort': {'interest_score': -1}},
            {'$limit': 200},
        ])

    @query('test', 1.0, 'Top reviews for one product (review collection, no join)')
    def _order_details(self):
        return MongoFindQuery('review',
            filter={
                'product_id': self._param_product_id(),
            },
            projection={
                '_id': 0,
                'review_id': 1,
                'customer_id': 1,
                'rating': 1,
                'title': 1,
                'body': 1,
                'helpful_votes': 1,
                'created_at': 1,
            },
            sort={
                'helpful_votes': -1,
                'created_at': -1,
            },
            limit=20,
        )
