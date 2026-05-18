from core.drivers import DriverType
from core.query import query, MongoQuery, MongoFindQuery, MongoAggregateQuery, ValueType
from ...common.edbt.query_registry import EdbtQueryRegistry

def export():
    return MongoEdbtQueryRegistry()

class MongoEdbtQueryRegistry(EdbtQueryRegistry[MongoQuery]):

    def __init__(self):
        super().__init__(DriverType.MONGO)

    def _param_customer_id(self):
        return self._param_int('customer_id', 1, self._counts.order)

    def _param_customer_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('customer_ids', self._counts.order, min_count, max_count)

    def _param_order_id(self):
        return self._param_int('order_id', 1, self._counts.order)

    def _param_order_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('order_ids', self._counts.order, min_count, max_count)

    def _param_country_code(self):
        return self._param_choice('country_code', COUNTRY_CODES)

    def _param_country_codes(self, min_count: int = 2, max_count: int = 5):
        return self._param('country_codes', lambda: self._convert_array(
            self._rng.sample(COUNTRY_CODES, self._rng_int(min_count, max_count)),
            ValueType.STRING,
        ))

    def _param_order_status(self):
        return self._param_choice('status', ORDER_STATUSES)

    def _param_order_statuses(self):
        return self._param('statuses', lambda: self._convert_array(
            self._rng.sample(ORDER_STATUSES, self._rng_int(1, 3)),
            ValueType.STRING,
        ))

    def _param_currency(self):
        return self._param_choice('currency', CURRENCIES)

    def _param_payment_method(self):
        return self._param_choice('payment_method', PAYMENT_METHODS)

    def _param_shipping_method(self):
        return self._param_choice('shipping_method', SHIPPING_METHODS)

    @query('edbt-0', 'Order history for a person (order, customer)')
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

    @query('edbt-1', 'Order details view (order, customer, order_item, product)')
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

    @query('edbt-2', 'How many times did this person bought these products? (order, customer, order_item)')
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

    @query('edbt-3', 'Seller daily revenue for last 30 days (order, order_item, product)')
    def _seller_daily_revenue(self):
        date = self._param_date_minus_days(30, 120)
        seller_ids = self._param_seller_ids(5, 50)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': date},
                'status': {'$in': ['paid', 'shipped']},
                'items.product.seller_id': {'$in': seller_ids},
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

    @query('edbt-4', 'Top catalog products inside categories by embedded product signals')
    def _top_catalog_products_by_category(self):
        category_ids = self._param_category_ids(10, 50)

        return MongoAggregateQuery('product', [
            {'$match': {
                'is_active': True,
                'categories.category.category_id': {'$in': category_ids},
            }},
            {'$addFields': {
                'review_count': {'$size': {'$ifNull': ['$reviews', []]}},
                'average_rating': {'$ifNull': [{'$avg': '$reviews.rating'}, 0]},
                'category_matches': {
                    '$size': {
                        '$filter': {
                            'input': {'$ifNull': ['$categories', []]},
                            'as': 'category',
                            'cond': {'$in': ['$$category.category.category_id', category_ids]},
                        },
                    },
                },
            }},
            {'$project': {
                '_id': 0,
                'product_id': 1,
                'title': 1,
                'price_cents': 1,
                'stock_qty': 1,
                'seller': {
                    'seller_id': '$seller.seller_id',
                    'country_code': '$seller.country_code',
                },
                'category_matches': 1,
                'review_count': 1,
                'average_rating': 1,
            }},
            {'$sort': {'average_rating': -1, 'review_count': -1, 'stock_qty': -1}},
            {'$limit': 200},
        ])

    @query('edbt-5', 'Customer spend buckets (order, customer)')
    def _customer_spend_buckets(self):
        person_ids = self._param_person_ids(100, 1000)

        return MongoAggregateQuery('order', [
            {'$match': {
                'customer.person_id': {'$in': person_ids},
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

    @query('edbt-6', 'Fraud-ish pattern (order, customer, order_item, product)')
    def _fraud_pattern(self):
        date = self._param_date_minus_days(1, 7)
        distinct_sellers_threshold = self._param_int('distinct_sellers_threshold', 2, 5)

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

    @query('edbt-7', 'Followers of a person from embedded follow edges')
    def _followers_for_person(self):
        person_id = self._param_person_id()

        return MongoAggregateQuery('person', [
            {'$match': {'follows.to_person.person_id': person_id}},
            {'$project': {
                '_id': 0,
                'person_id': 1,
                'name': 1,
                'country_code': 1,
                'is_active': 1,
                'matched_follows': {
                    '$filter': {
                        'input': '$follows',
                        'as': 'follow',
                        'cond': {'$eq': ['$$follow.to_person.person_id', person_id]},
                    },
                },
                'follows_count': {'$size': {'$ifNull': ['$follows', []]}},
            }},
            {'$unwind': '$matched_follows'},
            {'$sort': {'matched_follows.created_at': -1}},
            {'$limit': 200},
            {'$project': {
                'person_id': 1,
                'name': 1,
                'country_code': 1,
                'is_active': 1,
                'followed_at': '$matched_follows.created_at',
                'follows_count': 1,
            }},
        ])

    @query('edbt-8', 'Interest segment summary from embedded person interests')
    def _interest_segment_summary(self):
        category_ids = self._param_category_ids(10, 80)

        return MongoAggregateQuery('person', [
            {'$match': {
                'is_active': True,
                'interests.category.category_id': {'$in': category_ids},
            }},
            {'$unwind': '$interests'},
            {'$match': {
                'interests.category.category_id': {'$in': category_ids},
            }},
            {'$group': {
                '_id': {
                    'category_id': '$interests.category.category_id',
                    'country_code': '$country_code',
                },
                'people': {'$sum': 1},
                'avg_strength': {'$avg': '$interests.strength'},
                'recent_interest_at': {'$max': '$interests.created_at'},
            }},
            {'$project': {
                '_id': 0,
                'category_id': '$_id.category_id',
                'country_code': '$_id.country_code',
                'people': 1,
                'avg_strength': 1,
                'recent_interest_at': 1,
            }},
            {'$sort': {'people': -1, 'avg_strength': -1}},
            {'$limit': 200},
        ])

    @query('edbt-9', 'Product page read (product, seller, review)')
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

    @query('edbt-10', 'People also bought using shared orders (order, order_item)')
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

    @query('edbt-11', 'Recent order search by status and shipping country')
    def _recent_orders_by_status_country(self):
        return MongoFindQuery('order',
            filter={
                'ordered_at': {'$gte': self._param_date_minus_days(7, 90)},
                'status': self._param_order_status(),
                'shipping.country': self._param_country_code(),
            },
            projection={
                'order_id': 1,
                'ordered_at': 1,
                'status': 1,
                'total_cents': 1,
                'shipping': 1,
                '_id': 0,
            },
            sort={'ordered_at': -1},
            limit=self._param_limit(25, 4),
        )

    @query('edbt-12', 'Customer snapshot search by country and active flag')
    def _customer_snapshot_search(self):
        return MongoFindQuery('customer',
            filter={
                'country_code': self._param_country_code(),
                'is_active': True,
                'snapshot_at': {'$gte': self._param_date_minus_days(30, 180)},
            },
            projection={'customer_id': 1, 'person_id': 1, 'snapshot_at': 1, 'country_code': 1, '_id': 0},
            sort={'snapshot_at': -1},
            limit=self._param_limit(25, 4),
        )

    @query('edbt-13', 'Seller catalog browse from embedded product seller snapshot')
    def _seller_catalog_browse(self):
        return MongoFindQuery('product',
            filter={
                'seller.seller_id': self._param_seller_id(),
                'is_active': True,
            },
            projection={'product_id': 1, 'title': 1, 'price_cents': 1, 'stock_qty': 1, 'seller': 1, '_id': 0},
            sort={'price_cents': 1},
            limit=self._param_limit(20, 4),
        )

    @query('edbt-14', 'Category product browse with price and stock filters')
    def _category_product_browse(self):
        return MongoFindQuery('product',
            filter={
                'is_active': True,
                'categories.category.category_id': {'$in': self._param_category_ids(3, 20)},
                'stock_qty': {'$gt': self._param_int('stock_qty', 0, 2000)},
                'price_cents': {'$lte': self._param_int('price_cents', 1000, 20000)},
            },
            projection={'product_id': 1, 'title': 1, 'price_cents': 1, 'stock_qty': 1, 'categories': {'$slice': 3}, '_id': 0},
            sort={'stock_qty': -1},
            limit=self._param_limit(25, 4),
        )

    @query('edbt-15', 'Review distribution inside product categories')
    def _category_review_distribution(self):
        return MongoAggregateQuery('product', [
            {'$match': {
                'is_active': True,
                'categories.category.category_id': {'$in': self._param_category_ids(5, 30)},
            }},
            {'$unwind': '$reviews'},
            {'$group': {
                '_id': '$reviews.rating',
                'reviews': {'$sum': 1},
                'avg_helpful_votes': {'$avg': '$reviews.helpful_votes'},
            }},
            {'$sort': {'_id': -1}},
        ])

    @query('edbt-16', 'Product review timeline from review collection')
    def _product_review_timeline(self):
        return MongoAggregateQuery('review', [
            {'$match': {
                'product_id': {'$in': self._param_product_ids(10, 80)},
                'created_at': {'$gte': self._param_date_minus_days(30, 180)},
            }},
            {'$group': {
                '_id': {
                    'day': {'$dateTrunc': {'date': '$created_at', 'unit': 'day'}},
                    'rating': '$rating',
                },
                'reviews': {'$sum': 1},
                'avg_helpful_votes': {'$avg': '$helpful_votes'},
            }},
            {'$sort': {'_id.day': 1, '_id.rating': -1}},
        ])

    @query('edbt-17', 'Recent helpful reviews stream')
    def _recent_helpful_reviews(self):
        return MongoFindQuery('review',
            filter={
                'created_at': {'$gte': self._param_date_minus_days(7, 90)},
                'helpful_votes': {'$gte': self._param_int('helpful_votes', 5, 40)},
                'rating': {'$gte': self._param_int('rating', 3, 5)},
            },
            projection={'review_id': 1, 'product_id': 1, 'customer_id': 1, 'rating': 1, 'helpful_votes': 1, 'created_at': 1, '_id': 0},
            sort={'created_at': -1},
            limit=self._param_limit(50, 4),
        )

    @query('edbt-18', 'Seller activity by country')
    def _seller_activity_by_country(self):
        return MongoAggregateQuery('seller', [
            {'$match': {'country_code': {'$in': self._param_country_codes(2, 6)}}},
            {'$group': {
                '_id': {'country_code': '$country_code', 'is_active': '$is_active'},
                'sellers': {'$sum': 1},
            }},
            {'$sort': {'sellers': -1}},
        ])

    @query('edbt-19', 'Active person directory by country')
    def _active_person_directory(self):
        return MongoFindQuery('person',
            filter={
                'country_code': self._param_country_code(),
                'is_active': True,
            },
            projection={'person_id': 1, 'name': 1, 'country_code': 1, 'profile.lang': 1, '_id': 0},
            limit=self._param_limit(50, 5),
        )

    @query('edbt-20', 'People with strong interest in selected categories')
    def _people_with_strong_interest(self):
        return MongoFindQuery('person',
            filter={
                'interests': {
                    '$elemMatch': {
                        'category.category_id': {'$in': self._param_category_ids(5, 30)},
                        'strength': {'$gte': self._param_int('strength', 6, 10)},
                    },
                },
                'is_active': True,
            },
            projection={'person_id': 1, 'country_code': 1, 'interests': {'$slice': 6}, '_id': 0},
            limit=self._param_limit(50, 4),
        )

    @query('edbt-21', 'Follow graph summary by followed country')
    def _follow_graph_country_summary(self):
        return MongoAggregateQuery('person', [
            {'$match': {'country_code': self._param_country_code()}},
            {'$unwind': '$follows'},
            {'$group': {
                '_id': '$follows.to_person.country_code',
                'edges': {'$sum': 1},
                'active_targets': {'$sum': {'$cond': ['$follows.to_person.is_active', 1, 0]}},
            }},
            {'$sort': {'edges': -1}},
            {'$limit': 20},
        ])

    @query('edbt-22', 'Shipping and payment mix for recent orders')
    def _shipping_payment_mix(self):
        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': self._param_date_minus_days(7, 180)},
                'status': {'$in': self._param_order_statuses()},
            }},
            {'$group': {
                '_id': {
                    'shipping_country': '$shipping.country',
                    'shipping_method': '$shipping.method',
                    'payment_method': '$payment.method',
                },
                'orders': {'$sum': 1},
                'revenue_cents': {'$sum': '$total_cents'},
            }},
            {'$sort': {'orders': -1}},
            {'$limit': 100},
        ])

    @query('edbt-23', 'Product unit sales by day from embedded order items')
    def _product_unit_sales_by_day(self):
        product_ids = self._param_product_ids(10, 100)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': self._param_date_minus_days(7, 120)},
                'status': {'$in': ['paid', 'shipped']},
                'items.product.product_id': {'$in': product_ids},
            }},
            {'$unwind': '$items'},
            {'$match': {'items.product.product_id': {'$in': product_ids}}},
            {'$group': {
                '_id': {
                    'day': {'$dateTrunc': {'date': '$ordered_at', 'unit': 'day'}},
                    'product_id': '$items.product.product_id',
                },
                'units': {'$sum': '$items.quantity'},
                'revenue_cents': {'$sum': '$items.line_total_cents'},
            }},
            {'$sort': {'revenue_cents': -1}},
            {'$limit': 200},
        ])

    @query('edbt-24', 'Recent orders containing products from selected sellers')
    def _orders_by_embedded_product_seller(self):
        seller_ids = self._param_seller_ids(5, 50)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': self._param_date_minus_days(7, 90)},
                'status': {'$in': ['paid', 'shipped']},
                'items.product.seller_id': {'$in': seller_ids},
            }},
            {'$project': {
                '_id': 0,
                'order_id': 1,
                'ordered_at': 1,
                'customer.person_id': 1,
                'matching_items': {
                    '$filter': {
                        'input': '$items',
                        'as': 'item',
                        'cond': {'$in': ['$$item.product.seller_id', seller_ids]},
                    },
                },
            }},
            {'$sort': {'ordered_at': -1}},
            {'$limit': 200},
        ])

    @query('edbt-25', 'Inventory by seller country and currency')
    def _inventory_by_seller_country_currency(self):
        return MongoAggregateQuery('product', [
            {'$match': {
                'is_active': True,
                'seller.country_code': {'$in': self._param_country_codes(2, 6)},
                'stock_qty': {'$gt': 0},
            }},
            {'$group': {
                '_id': {
                    'country_code': '$seller.country_code',
                    'currency': '$currency',
                },
                'products': {'$sum': 1},
                'stock_qty': {'$sum': '$stock_qty'},
                'avg_price_cents': {'$avg': '$price_cents'},
            }},
            {'$sort': {'stock_qty': -1}},
        ])

    @query('edbt-26', 'Category tree prefix browse')
    def _category_tree_prefix_browse(self):
        return MongoFindQuery('category',
            filter={'path': {'$regex': self._param_choice('path_prefix', ['^/cat1', '^/cat2', '^/cat3', '^/cat4', '^/cat5'])}},
            projection={'category_id': 1, 'name': 1, 'path': 1, '_id': 0},
            sort={'path': 1},
            limit=self._param_limit(20, 4),
        )

    @query('edbt-27', 'Customer snapshots for selected people')
    def _customer_snapshots_for_people(self):
        return MongoFindQuery('customer',
            filter={'person_id': {'$in': self._param_person_ids(10, 80)}},
            projection={'customer_id': 1, 'person_id': 1, 'snapshot_at': 1, 'country_code': 1, '_id': 0},
            sort={'snapshot_at': -1},
            limit=self._param_limit(50, 4),
        )

    @query('edbt-28', 'Customer review history')
    def _customer_review_history(self):
        return MongoFindQuery('review',
            filter={'customer_id': {'$in': self._param_customer_ids(10, 80)}},
            projection={'review_id': 1, 'product_id': 1, 'customer_id': 1, 'rating': 1, 'created_at': 1, '_id': 0},
            sort={'created_at': -1},
            limit=self._param_limit(50, 4),
        )

    @query('edbt-29', 'Hot products by recent review volume')
    def _hot_products_by_recent_reviews(self):
        product_ids = self._param_product_ids(100, 1000)

        return MongoAggregateQuery('review', [
            {'$match': {
                'product_id': {'$in': product_ids},
                'created_at': {'$gte': self._param_date_minus_days(7, 180)},
            }},
            {'$group': {
                '_id': '$product_id',
                'reviews': {'$sum': 1},
                'avg_rating': {'$avg': '$rating'},
                'avg_helpful_votes': {'$avg': '$helpful_votes'},
            }},
            {'$sort': {'reviews': -1, 'avg_rating': -1}},
            {'$limit': 200},
        ])

    @query('edbt-30', 'Basket size distribution for recent orders')
    def _basket_size_distribution(self):
        product_ids = self._param_product_ids(10, 80)

        return MongoAggregateQuery('order', [
            {'$match': {
                'ordered_at': {'$gte': self._param_date_minus_days(7, 180)},
                'status': {'$in': ['paid', 'shipped']},
                'items.product.product_id': {'$in': product_ids},
            }},
            {'$project': {
                'status': 1,
                'item_count': {'$size': {'$ifNull': ['$items', []]}},
                'total_cents': 1,
            }},
            {'$bucket': {
                'groupBy': '$item_count',
                'boundaries': [0, 2, 4, 8, 16],
                'default': '16+',
                'output': {
                    'orders': {'$sum': 1},
                    'avg_total_cents': {'$avg': '$total_cents'},
                },
            }},
        ])

    @query('edbt-31', 'High-value orders by payment and shipping method')
    def _high_value_orders_by_payment_shipping(self):
        return MongoFindQuery('order',
            filter={
                'ordered_at': {'$gte': self._param_date_minus_days(7, 180)},
                'currency': self._param_currency(),
                'payment.method': self._param_payment_method(),
                'shipping.method': self._param_shipping_method(),
                'total_cents': {'$gte': self._param_int('total_cents', 5_000, 50_000)},
            },
            projection={
                'order_id': 1,
                'ordered_at': 1,
                'status': 1,
                'total_cents': 1,
                'currency': 1,
                'shipping': 1,
                'payment': 1,
                '_id': 0,
            },
            sort={'total_cents': -1},
            limit=self._param_limit(20, 4),
        )


COUNTRY_CODES = ['AU', 'BR', 'CA', 'CZ', 'DE', 'ES', 'FR', 'GB', 'IN', 'IT', 'NL', 'PL', 'SE', 'US']
ORDER_STATUSES = ['paid', 'shipped', 'canceled', 'refunded']
CURRENCIES = ['AUD', 'BRL', 'CAD', 'EUR', 'GBP', 'INR', 'USD']
PAYMENT_METHODS = ['card', 'paypal', 'bank']
SHIPPING_METHODS = ['standard', 'express']
