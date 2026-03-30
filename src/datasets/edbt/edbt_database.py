from common.config import DatasetName
from common.database import Database
from common.drivers import DriverType
from common.query_registry import TQuery
from datasets.edbt.data_generator import EdbtDataGenerator

class EdbtDatabase(Database[TQuery]):

    def __init__(self, driver: DriverType, scale: float):
        super().__init__(DatasetName.EDBT, driver, scale)
        generator = EdbtDataGenerator(scale)
        self._counts = generator.generate_counts()

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
