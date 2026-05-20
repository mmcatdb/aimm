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
