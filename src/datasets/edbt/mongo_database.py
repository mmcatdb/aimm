from datetime import datetime, timedelta

from common.query_registry import query
from common.drivers import DriverType
from common.database import MongoQuery, MongoFindQuery, MongoAggregateQuery
from datasets.edbt.edbt_database import EdbtDatabase


class EdbtMongoDatabase(EdbtDatabase[MongoQuery]):

    def __init__(self, scale: float):
        super().__init__(DriverType.MONGO, scale)

    def _param_customer_id(self):
        return self._param_int('customer_id', 1, 30000)

    @query('test', 1.0, 'Order history for a person (order, customer)')
    def _order_history_for_person(self):
        return MongoFindQuery('order',
            filter={'customer.person_id': self._param_person_id()},
            projection={
                'order_id': 1,
                'ordered_at': 1,
                'status': 1,
                'total_cents': 1,
                'currency': 1,
                '_id': 0,
            },
            sort={'ordered_at': -1},
        )

    @query('test', 1.0, 'Order details view (order, customer, order_item, product)')
    def _order_details(self):
        return MongoAggregateQuery('order', [
            {'$match': {'customer.person_id': self._param_person_id()}},
            {'$unwind': '$items'},
            {'$project': {
                '_id': 0,
                'order_id': '$order_id',
                'ordered_at': '$ordered_at',
                'status': '$status',
                'order_item_id': '$items.order_item_id',
                'product_id': '$items.product.product_id',
                'product_title': {'$ifNull': ['$items.product.title', None]},
                'unit_price_cents': '$items.unit_price_cents',
                'quantity': '$items.quantity',
                'line_total_cents': '$items.line_total_cents',
            }},
            {'$sort': {'order_item_id': 1}},
        ])

    @query('test', 1.0, 'How many times did this person bought these products? (order, customer, order_item)')
    def _product_purchases_for_person(self):
        person_ids = self._param_person_ids(100, 1000)
        product_ids = self._param_product_ids(100, 1000)

        return MongoAggregateQuery('order', [
            {'$match': {
                'customer.person_id': {'$in': person_ids},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$match': {'items.product.product_id': {'$in': product_ids}}},
            {'$count': 'count'},
        ])

    @query('test', 1.0, 'Seller daily revenue for last 30 days (order, order_item, product)')
    def _seller_daily_revenue(self):
        date = self._param_date_minus_days(30, 120)
        seller_ids = self._param_seller_ids(100, 1000)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': date},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$match': {'items.product.seller_id': {'$in': seller_ids}}},
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

    @query('test', 1.0, 'Top products by revenue inside one category, last 7-30 days (order, order_item, product, has_category)')
    def _top_products_by_revenue(self):
        date = self._param_date_minus_days(7, 30)
        category_ids = self._param_category_ids(10, 50)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': date},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$lookup': {
                'from': 'product',
                'let': {'pid': '$items.product.product_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$product_id', '$$pid']}}},
                    {'$match': {'categories.category.category_id': {'$in': category_ids}}},
                ],
                'as': 'product'
            }},
            {'$unwind': '$product'},
            {'$group': {
                '_id': '$items.product.product_id',
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
            {'$limit': 200},
        ])

    @query('test', 1.0, 'Customer spend buckets (order, customer)')
    def _customer_spend_buckets(self):
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': self._param_date_minus_days(30, 180)},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$group': {
                '_id': '$customer.person_id',
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

    @query('test', 1.0, 'Fraud-ish pattern (order, customer, order_item, product)')
    def _fraud_pattern(self):
        date = self._param_date_minus_days(1, 7)
        distinct_sellers_threshold = self._param_int('distinct_sellers_threshold', 10, 1000)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': date},
                'status': {'$in': ['paid', 'shipped']},
            }},
            {'$unwind': '$items'},
            {'$group': {
                '_id': '$customer.person_id',
                'distinct_products_set': {'$addToSet': '$items.product.seller_id'},
                'order_ids': {'$addToSet': '$order_id'},
            }},
            {'$project': {
                '_id': 0,
                'person_id': '$_id',
                'distinct_sellers': {'$size': '$distinct_products_set'},
                'order': {'$size': '$order_ids'},
            }},
            {'$match': {'distinct_sellers': {'$gte': distinct_sellers_threshold}}},
            {'$sort': {'distinct_sellers': -1}},
            {'$limit': 200},
        ])

    @query('test', 1.0, 'Who should I follow? (follows)')
    def _who_to_follow(self):
        person_id = self._param_person_id()

        return MongoAggregateQuery('person', [
            {'$match': {'follows.to_person.person_id': person_id}},
            {'$project': {'from_id': '$person_id'}},
            {'$lookup': {
                'from': 'person',
                'let': {'f1_id': '$from_id'},
                'pipeline': [
                    {'$match': {'$expr': {
                        '$in': ['$$f1_id', '$follows.to_person.person_id'],
                    }}},
                    {'$project': {
                        'person_id': 1,
                        'follows': 1,
                    }},
                ],
                'as': 'f2',
            }},
            {'$unwind': '$f2'},
            {'$match': {'f2.person_id': {'$ne': person_id}}},
            {'$match': {'f2.follows': {
                '$not': {'$elemMatch': {'to_person.person_id': person_id}},
            }}},
            {'$group': {
                '_id': '$f2.person_id',
                'paths': {'$sum': 1},
            }},
            {'$sort': {'paths': -1}},
            {'$limit': 200},
            {'$project': {
                '_id': 0,
                'person_id': '$_id',
                'paths': 1,
            }},
        ])

    @query('test', 1.0, 'Personalized feed candidates (product, has_category, has_interest)')
    def _personalized_feed_candidates(self):
        return MongoAggregateQuery('person', [
            {'$match': {'person_id': {'$in': self._param_person_ids(2, 20)}}},
            {'$unwind': '$interests'},
            {'$lookup': {
                'from': 'product',
                'let': {'cat_id': '$interests.category.category_id'},
                'pipeline': [
                    {'$match': {
                        '$expr': {'$and': [
                            {'$eq': ['$is_active', True]},
                            {'$in': ['$$cat_id', '$categories.category.category_id']},
                        ]},
                    }},
                    {'$project': {
                        'product_id': 1,
                    }},
                ],
                'as': 'products',
            }},
            {'$unwind': '$products'},
            {'$group': {
                '_id': '$products.product_id',
                'interest_score': {'$sum': '$interests.strength'},
            }},
            {'$sort': {'interest_score': -1 }},
            {'$limit': 200},
            {'$project': {
                '_id': 0,
                'product_id': '$_id',
                'interest_score': 1,
            }},
        ])

    @query('test', 1.0, 'Product page read (product, seller, review)')
    def _product_page_read(self):
        return MongoAggregateQuery('product', [
            {'$match': {
                'product_id': self._param_product_id(),
                'is_active': True,
            }},
            {'$addFields': {
                'rating_summary': {
                    'avg': {
                        '$ifNull': [{'$avg': '$reviews.rating'}, 0]
                    },
                    'count': {
                        '$size': {'$ifNull': ['$reviews', []]}
                    },
                },
                'top_reviews': {
                    '$slice': [{
                        '$sortArray': {
                        'input': {'$ifNull': ['$reviews', []]},
                        'sortBy': {'helpful_votes': -1, 'created_at': -1}
                        }
                    }, 50],
                },
            }},
            {'$project': {
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'price_cents': 1,
                'currency': 1,
                'stock_qty': 1,
                'seller': {
                    'seller_id': '$seller.seller_id',
                    'display_name': '$seller.display_name'
                },
                'rating_summary': 1,
                'top_reviews': '$top_reviews'
            }},
        ])

    @query('test', 1.0, 'People also bought using shared orders (order, order_item)')
    def _people_also_bought(self):
        product_ids = self._param_product_ids(10, 50)

        return MongoAggregateQuery('order', [
            {'$match': {
                'status': {'$in': ['paid', 'shipped']},
                'items.product.product_id': {'$in': product_ids},
            }},
            {'$unwind': '$items'},
            {'$match': {
                'items.product.product_id': {'$nin': product_ids},
            }},
            {'$group': {
                '_id': '$items.product.product_id',
                'co_buy': {'$sum': 1},
            }},
            {'$sort': {'co_buy': -1}},
            {'$limit': 20},
        ])
