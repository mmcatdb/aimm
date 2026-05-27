from abc import abstractmethod
from typing_extensions import override
from core.drivers import DriverType
from core.query import SchemaName, QueryRegistry, TQuery, ValueType
from .data_generator import EdbtDataGenerator

EDBT_SCHEMA: SchemaName = 'edbt'

class EdbtQueryRegistry(QueryRegistry[TQuery]):

    def __init__(self, driver: DriverType):
        super().__init__(driver, EDBT_SCHEMA)

    @override
    def _setup_cache(self):
        generator = EdbtDataGenerator()
        self._counts = generator.generate_counts(self._scale)

    # Ids

    def _param_person_id(self):
        return self._param_int('person_id', 1, self._counts.person)

    def _param_person_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('person_ids', self._counts.person, min_count, max_count)

    def _param_product_id(self):
        return self._param_int('product_id', 1, self._counts.product)

    def _param_product_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('product_ids', self._counts.product, min_count, max_count)

    def _param_seller_id(self):
        return self._param_int('seller_id', 1, self._counts.seller)

    def _param_seller_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('seller_ids', self._counts.seller, min_count, max_count)

    def _param_category_id(self):
        return self._param_int('category_id', 1, self._counts.category)

    def _param_category_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('category_ids', self._counts.category, min_count, max_count)

    def _param_customer_id(self):
        return self._param_int('customer_id', 1, self._counts.order)

    def _param_customer_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('customer_ids', self._counts.order, min_count, max_count)

    def _param_order_id(self):
        return self._param_int('order_id', 1, self._counts.order)

    def _param_order_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('order_ids', self._counts.order, min_count, max_count)

    # Other

    def _param_country_code(self):
        return self._param_choice('country_code', EdbtDataGenerator.COUNTRY_CODES)

    def _param_country_codes(self, min_count: int = 2, max_count: int = 5):
        return self._param('country_codes', lambda: self._convert_array(self._rng.sample(EdbtDataGenerator.COUNTRY_CODES, self._rng_int(min_count, max_count)), ValueType.STRING))

    def _param_order_status(self):
        return self._param_choice('status', EdbtDataGenerator.ORDER_STATUSES)

    def _param_order_statuses(self):
        return self._param('statuses', lambda: self._convert_array(self._rng.sample(EdbtDataGenerator.ORDER_STATUSES, self._rng_int(1, 3)), ValueType.STRING))

    def _param_currency(self):
        return self._param_choice('currency', EdbtDataGenerator.CURRENCIES)

    def _param_payment_method(self):
        return self._param_choice('payment_method', EdbtDataGenerator.PAYMENT_METHODS)

    def _param_shipping_method(self):
        return self._param_choice('shipping_method', EdbtDataGenerator.SHIPPING_METHODS)

    # Common MCTS queries

    def _register_queries(self):
        # OLTP focused (mostly Postgres)

        self._query('mcts-0', 'Order history for a person (order, customer)', lambda s: s._order_history_for_person(
            person_id=s._param_person_id(),
        ))

        self._query('mcts-1', 'Order details view (order, customer, order_item, product)', lambda s: s._order_details(
            person_id=self._param_person_id(),
        ))

        self._query('mcts-2', 'How many times did this person bought these products? (order, customer, order_item)', lambda s: s._product_purchases_for_person(
            person_ids=self._param_person_ids(100, 1000),
            product_ids=self._param_product_ids(100, 1000),
        ))

        # OLAP focused (Postgres)

        self._query('mcts-3', 'Seller daily revenue for last 30 days (order, order_item, product)', lambda s: s._seller_daily_revenue(
            date=self._param_date_minus_days(30, 120),
            seller_ids=self._param_seller_ids(100, 1000),
        ))

        self._query('mcts-4', 'Top products by revenue inside one category, last 7-30 days (order, order_item, product, has_category)', lambda s: s._top_products_by_revenue(
            date=self._param_date_minus_days(7, 30),
            category_ids=self._param_category_ids(10, 50),
        ))

        self._query('mcts-5', 'Customer spend buckets (order, customer)', lambda s: s._customer_spend_buckets(
            date=self._param_date_minus_days(30, 180),
        ))

        self._query('mcts-6', 'Fraud-ish pattern (order, customer, order_item, product)', lambda s: s._fraud_pattern(
            date=self._param_date_minus_days(1, 7),
            distinct_sellers_threshold=self._param_int('distinct_sellers_threshold', 10, 1000),
        ))

        self._query('mcts-7', 'Who should I follow? (follows)', lambda s: s._who_to_follow(
            person_id=self._param_person_id(),
        ))

        self._query('mcts-8', 'Personalized feed candidates (product, has_category, has_interest)', lambda s: s._personalized_feed_candidates(
            person_ids=self._param_person_ids(2, 20),
        ))

        # Document focused (MongoDB)
        # These are built to avoid joins at read time. Put "product page bundle" in one document.

        self._query('mcts-9', 'Product page read (product, seller, review)', lambda s: s._product_page_read(
            product_id=self._param_product_id(),
        ))

        self._query('mcts-10', 'People also bought using shared orders (order, order_item)', lambda s: s._people_also_bought(
            product_ids=self._param_product_ids(10, 50),
        ))

    # Abstract methods for common MCTS queries

    @abstractmethod
    def _order_history_for_person(self, person_id) -> TQuery: ...

    @abstractmethod
    def _order_details(self, person_id) -> TQuery: ...

    @abstractmethod
    def _product_purchases_for_person(self, person_ids, product_ids) -> TQuery: ...

    @abstractmethod
    def _seller_daily_revenue(self, date, seller_ids) -> TQuery: ...

    @abstractmethod
    def _top_products_by_revenue(self, date, category_ids) -> TQuery: ...

    @abstractmethod
    def _customer_spend_buckets(self, date) -> TQuery: ...

    @abstractmethod
    def _fraud_pattern(self, date, distinct_sellers_threshold) -> TQuery: ...

    @abstractmethod
    def _who_to_follow(self, person_id) -> TQuery: ...

    @abstractmethod
    def _personalized_feed_candidates(self, person_ids) -> TQuery: ...

    @abstractmethod
    def _product_page_read(self, product_id) -> TQuery: ...

    @abstractmethod
    def _people_also_bought(self, product_ids) -> TQuery: ...
