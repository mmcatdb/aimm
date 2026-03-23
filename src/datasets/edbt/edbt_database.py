from common.config import DatasetName
from common.database import Database
from common.drivers import DriverType
from common.query_registry import TQuery

class EdbtDatabase(Database[TQuery]):

    def __init__(self, driver: DriverType):
        super().__init__(DatasetName.EDBT, driver)

    # Ids

    def _param_person_id(self):
        return self._param_int('person_id', 1, MAX_PERSON_ID)

    def _param_product_id(self):
        return self._param_int('product_id', 1, MAX_PRODUCT_ID)

    def _param_product_ids(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('product_ids', MAX_PRODUCT_ID, min_count, max_count)

    def _param_seller_id(self):
        return self._param_int('seller_id', 1, MAX_SELLER_ID)

    def _param_category_id(self):
        return self._param_int('category_id', 1, MAX_CATEGORY_ID)


# FIXME this
MAX_PERSON_ID = 30000
MAX_PRODUCT_ID = 20000
MAX_SELLER_ID = 10000
MAX_CATEGORY_ID = 1000
