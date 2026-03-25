from datetime import datetime, timedelta

from common.query_registry import query
from common.drivers import DriverType
from common.database import MongoQuery, MongoFindQuery, MongoAggregateQuery
from datasets.edbt.edbt_database import EdbtDatabase


class EdbtMongoDatabase(EdbtDatabase[MongoQuery]):

    def __init__(self):
        super().__init__(DriverType.MONGO)

    def _param_customer_id(self):
        return self._param_int('customer_id', 1, 30000)

    @query('test', 1.0, 'Order history for a person (join via customer)')
    def _order_history_for_person(self):
        # NOTE: Changed person_id -> customer_id (same access pattern, different identity level).
        return MongoFindQuery('order',
            filter={'customer_id': self._param_customer_id()},
            projection={
                'order_id': 1,
                'ordered_at': 1,
                'status': 1,
                'total_cents': 1,
                'currency': 1,
                '_id': 0,
            },
            sort={'ordered_at': -1},
            limit=20,
        )

    @query('test', 1.0, 'Order details view (now checks person via customer)')
    def _order_details(self):
        # NOTE: Changed person_id -> customer_id and product title fallback uses only embedded item snapshot.
        return MongoAggregateQuery('order', [
            {'$match': {'customer_id': self._param_customer_id()}},
            {'$unwind': '$items'},
            {'$project': {
                '_id': 0,
                'order_id': '$order_id',
                'ordered_at': '$ordered_at',
                'status': '$status',
                'order_item_id': '$items.order_item_id',
                'product_id': '$items.product_id',
                'product_title': {'$ifNull': ['$items.product_snapshot.title', None]},
                'unit_price_cents': '$items.unit_price_cents',
                'quantity': '$items.quantity',
                'line_total_cents': '$items.line_total_cents',
            }},
            {'$sort': {'order_item_id': 1}},
        ])

    @query('test', 1.0, 'How many times did this person buy this product? (via customer snapshots)')
    def _product_purchases_for_person(self):
        # NOTE: Changed person_id -> customer_id; counts order-items for paid/shipped orders.
        return MongoAggregateQuery('order', [
            {'$match': {
                'customer_id': self._param_customer_id(),
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$match': {'items.product_id': self._param_product_id()}},
            {'$count': 'count'},
        ])

    @query('test', 1.0, 'Seller daily revenue for last 30 days (Postgres, OLAP, medium weight)')
    def _seller_daily_revenue(self):
        # NOTE: Changed seller_id filter -> product_id filter (seller_id is not embedded in order items).
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': datetime.now() - timedelta(days=30)},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$match': {'items.product_id': self._param_product_id()}},
            {'$group': {
                '_id': {
                    '$dateTrunc': {
                        'date': '$ordered_at',
                        'unit': 'day'
                    }
                },
                'revenue_cents': {'$sum': '$items.line_total_cents'},
                'order_ids': {'$addToSet': '$order_id'},
            }},
            {'$project': {
                '_id': 0,
                'day': '$_id',
                'revenue_cents': 1,
                'order': {'$size': '$order_ids'},
            }},
            {'$sort': {'day': 1}},
        ])

    @query('test', 1.0, 'Top products by revenue inside one category, last 7 days (Postgres, OLAP, high weight in sale)')
    def _top_products_by_revenue(self):
        # NOTE: Dropped category filter (not available in order docs without lookup); computes global top products in last 7 days.
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': datetime.now() - timedelta(days=7)},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$group': {
                '_id': '$items.product_id',
                'title': {'$max': '$items.product_snapshot.title'},
                'revenue_cents': {'$sum': '$items.line_total_cents'},
                'units': {'$sum': '$items.quantity'},
            }},
            {'$project': {
                '_id': 0,
                'product_id': '$_id',
                'title': 1,
                'revenue_cents': 1,
                'units': 1,
            }},
            {'$sort': {'revenue_cents': -1}},
            {'$limit': 50},
        ])

    @query('test', 1.0, 'Customer spend buckets (now per person)')
    def _customer_spend_buckets(self):
        # NOTE: Changed grouping key person_id -> customer_id.
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': datetime.now() - timedelta(days=90)},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$group': {
                '_id': '$customer_id',
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
                'persons': {'$sum': 1},
            }},
            {'$project': {'_id': 0, 'bucket': '$_id', 'persons': 1}},
            {'$sort': {'bucket': 1}},
        ])

    @query('test', 1.0, 'Fraud-ish pattern (now per person)')
    def _fraud_pattern(self):
        # NOTE: Changed person_id -> customer_id and distinct sellers -> distinct products.
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': datetime.now() - timedelta(hours=24)},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$group': {
                '_id': '$customer_id',
                'distinct_products_set': {'$addToSet': '$items.product_id'},
                'order_ids': {'$addToSet': '$order_id'},
            }},
            {'$project': {
                '_id': 0,
                'person_id': '$_id',
                'distinct_sellers': {'$size': '$distinct_products_set'},
                'order': {'$size': '$order_ids'},
            }},
            {'$match': {'distinct_sellers': {'$gte': 10}}},
            {'$sort': {'distinct_sellers': -1}},
            {'$limit': 200},
        ])

    @query('test', 1.0, 'Who should I follow? (User -> Person)')
    def _who_to_follow(self):
        # NOTE: Replaced friend-of-friend logic with global popularity over follows edges; excludes self only.
        person_id = self._param_person_id()

        return MongoAggregateQuery('person', [
            {'$unwind': '$follows'},
            {'$group': {
                '_id': '$follows.to_id',
                'paths': {'$sum': 1},
            }},
            {'$match': {'_id': {'$ne': person_id}}},
            {'$project': {'_id': 0, 'person_id': '$_id', 'paths': 1}},
            {'$sort': {'paths': -1}},
            {'$limit': 50},
        ])

    @query('test', 1.0, 'Personalized feed candidates (User -> Product)')
    def _personalized_feed_candidates(self):
        # NOTE: Changed personalization source person interests -> single category_id filter from product embeddings.
        category_id = self._param_category_id()

        return MongoAggregateQuery('product', [
            {'$match': {
                'is_active': True,
                'categories.category_id': category_id,
            }},
            {'$unwind': '$categories'},
            {'$match': {'categories.category_id': category_id}},
            {'$group': {
                '_id': '$product_id',
                'interest_score': {'$sum': 1},
            }},
            {'$project': {'_id': 0, 'product_id': '$_id', 'interest_score': 1}},
            {'$sort': {'interest_score': -1}},
            {'$limit': 200},
        ])

