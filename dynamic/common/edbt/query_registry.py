from typing_extensions import override
from core.drivers import DriverType
from core.query import SchemaName, QueryRegistry, TQuery
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
