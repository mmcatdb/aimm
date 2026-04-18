from core.drivers import DriverType
from core.query import SchemaName, QueryRegistry, TQuery, ValueType

TPCH_SCHEMA: SchemaName = 'tpch'

class TpchQueryRegistry(QueryRegistry[TQuery]):

    def __init__(self, driver: DriverType):
        super().__init__(driver, TPCH_SCHEMA)

    def _param_date(self, start_year=1992, end_year=1998):
        return self._param('date', lambda: self._convert_date(self._rng_date(start_year=start_year, end_year=end_year)))

    # Ids

    def _param_custkey(self):
        return self._param_int('custkey', 1, MAX_CUSTOMER_ID)

    def _param_orderkey(self):
        return self._param_int('orderkey', 1, MAX_ORDER_ID)

    def _param_orderkeys(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('orderkeys', MAX_ORDER_ID, min_count, max_count)

    def _param_partkey(self):
        return self._param_int('partkey', 1, MAX_PART_ID)

    def _param_partkeys(self, min_count: int, max_count: int | None = None):
        return self._param_int_array('partkeys', MAX_PART_ID, min_count, max_count)

    def _param_nationkey(self):
        return self._param_int('nationkey', 0, 24)

    def _param_suppkey(self):
        return self._param_int('suppkey', 1, MAX_SUPPLIER_ID)

    # Names

    def _param_customer(self):
        return self._param('customer', lambda: f'Customer#{self._rng_int(1, MAX_CUSTOMER_ID):09d}')

    def _param_supplier(self):
        return self._param('supplier', lambda: f'Supplier#{self._rng_int(1, MAX_SUPPLIER_ID):09d}')

    def _param_brand(self, name = 'brand'):
        return self._param(name, lambda: f'Brand#{self._rng_int(1, 5)}{self._rng_int(1, 5)}')

    def _param_brands(self, name = 'brands', count = 20):
        return self._param(name, lambda: self._convert_array([self._param_brand() for _ in range(count)], ValueType.STRING))

    # Other

    def _param_shipmodes(self, count = 2):
        return self._param('shipmodes', lambda: self._convert_array(self._rng.sample(SHIPMODES, count), ValueType.STRING))

    def _param_region(self, name = 'region', exclude: str | None = None):
        regions = REGIONS if exclude is None else [r for r in REGIONS if r != exclude]
        return self._param_choice(name, regions)

    def _param_nation(self):
        return self._param_choice('nation', NATIONS)

    def _param_part_name_word(self):
        return self._param_choice('part_name_word', PART_NAME_WORDS)

    def _param_order_status(self):
        return self._param_choice('order_status', ORDER_STATUSES)

    def _param_order_priority(self, first_n: int | None = None):
        priorities = ORDER_PRIORITIES if first_n is None else ORDER_PRIORITIES[:first_n]
        return self._param_choice('order_priority', priorities)

    def _param_line_status(self):
        return self._param_choice('line_status', LINE_STATUSES)

    def _param_part_type(self):
        return self._param_choice('part_type', PART_TYPES)

    def _param_containers(self, count = 3):
        return self._param('containers', lambda: self._convert_array(self._rng.sample(CONTAINERS, count), ValueType.STRING))

    def _param_segment(self):
        return self._param_choice('segment', SEGMENTS)

MAX_CUSTOMER_ID = 30000
MAX_ORDER_ID = 1200000
MAX_PART_ID = 40000
MAX_SUPPLIER_ID = 2000

SHIPMODES = ['MAIL', 'SHIP', 'AIR', 'TRUCK', 'RAIL', 'FOB', 'REG AIR']
REGIONS = ['ASIA', 'AMERICA', 'EUROPE', 'MIDDLE EAST', 'AFRICA']
NATIONS = ['ALGERIA', 'ARGENTINA', 'BRAZIL', 'CANADA', 'EGYPT', 'ETHIOPIA', 'FRANCE', 'GERMANY', 'INDIA', 'INDONESIA', 'IRAN', 'IRAQ', 'JAPAN', 'JORDAN', 'KENYA', 'MOROCCO', 'MOZAMBIQUE', 'PERU', 'CHINA', 'ROMANIA', 'SAUDI ARABIA', 'VIETNAM', 'RUSSIA', 'UNITED KINGDOM', 'UNITED STATES']
PART_NAME_WORDS = [ 'almond', 'antique', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanched', 'blue', 'blush', 'brown', 'burlywood', 'burnished', 'chartreuse', 'chiffon', 'chocolate', 'coral', 'cornflower', 'cornsilk', 'cream', 'cyan', 'dark', 'deep', 'dim', 'dodger', 'drab', 'firebrick', 'floral', 'forest', 'frosted', 'gainsboro', 'ghost', 'goldenrod', 'green', 'grey', 'honeydew', 'hot', 'indian', 'ivory', 'khaki', 'lace', 'lavender', 'lawn', 'lemon', 'light', 'lime', 'linen', 'magenta', 'maroon', 'medium', 'metallic', 'midnight', 'mint', 'misty', 'moccasin', 'navajo', 'navy', 'olive', 'orange', 'orchid', 'pale', 'papaya', 'peach', 'peru', 'pink', 'plum', 'powder', 'puff', 'purple', 'red', 'rose', 'rosy', 'royal', 'saddle', 'salmon', 'sandy', 'seashell', 'sienna', 'sky', 'slate', 'smoke', 'snow', 'spring', 'steel', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'yellow' ]
ORDER_STATUSES = ['F', 'O', 'P']
ORDER_PRIORITIES = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']
LINE_STATUSES = ['F', 'O']
PART_TYPES = [ 'STANDARD ANODIZED', 'STANDARD BURNISHED', 'STANDARD PLATED', 'STANDARD POLISHED', 'STANDARD BRUSHED', 'SMALL ANODIZED', 'SMALL BURNISHED', 'SMALL PLATED', 'SMALL POLISHED', 'SMALL BRUSHED', 'MEDIUM ANODIZED', 'MEDIUM BURNISHED', 'MEDIUM PLATED', 'MEDIUM POLISHED', 'MEDIUM BRUSHED', 'LARGE ANODIZED', 'LARGE BURNISHED', 'LARGE PLATED', 'LARGE POLISHED', 'LARGE BRUSHED', 'ECONOMY ANODIZED', 'ECONOMY BURNISHED', 'ECONOMY PLATED', 'ECONOMY POLISHED', 'ECONOMY BRUSHED', 'PROMO ANODIZED', 'PROMO BURNISHED', 'PROMO PLATED', 'PROMO POLISHED', 'PROMO BRUSHED' ]
CONTAINERS = ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG', 'MED BAG', 'MED BOX', 'MED PKG', 'MED PACK', 'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG']
SEGMENTS = ['BUILDING', 'AUTOMOBILE', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
