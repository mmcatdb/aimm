from collections.abc import Callable
from typing_extensions import override
from core.drivers import DriverType
from core.query import SchemaName, QueryRegistry, TQuery
from .data_generator import ArtDataGenerator

ART_SCHEMA: SchemaName = 'art'

class ArtQueryRegistry(QueryRegistry[TQuery]):
    """
    Base registry for the ART schema.
    Provides common parameter helpers used by driver-specific subclasses.
    """

    def __init__(self, driver: DriverType):
        super().__init__(driver, ART_SCHEMA)

    @override
    def _setup_cache(self):
        generator = ArtDataGenerator()
        self._counts = generator.generate_counts(self._scale)

    # ID parameters

    def _param_node_id(self, name: str = 'node_id'):
        return self._param_int(name, 1, self._counts.node)

    def _param_seed(self, key: str):
        """Generates a fresh ID that is guaranteed to not exist in the database.
        The range is (count, 2*count] for the given ArtCounts key, and the same ID is never generated twice.
        """
        def generate():
            count = getattr(self._counts, key)
            return self._rng_unique_int(key, count + 1, count * 2)

        return self._param(f'seed_{key}', generate)

    def _param_node_ids(self, min_c: int, max_c: int | None = None):
        return self._param_int_array('node_ids', self._counts.node, min_c, max_c)

    def _param_grp_id(self, name: str = 'grp_id'):
        return self._param_int(name, 1, self._counts.grp)

    def _param_grp_ids(self, min_c: int, max_c: int | None = None):
        return self._param_int_array('grp_ids', self._counts.grp, min_c, max_c)

    def _param_doc_id(self):
        return self._param_int('doc_id', 1, self._counts.doc)

    def _param_log_id(self):
        return self._param_int('log_id', 1, self._counts.log)

    def _param_measure_id(self):
        return self._param_int('measure_id', 1, self._counts.measure)

    def _param_event_id(self):
        return self._param_int('event_id', 1, self._counts.event_log)

    # Value / attribute parameters

    def _param_tag(self):
        """Random tag in the range T0000-T{n_tag-1}.  Data is Zipf; queries are uniform."""
        return self._param('tag', lambda: f'T{self._rng_int(0, self._counts.n_tag - 1):04d}')

    def _param_val_int(self):
        return self._param_int('val_int', 1, 1000)

    def _param_status(self):
        """node.status  0=active  1=inactive  2=pending  3=banned  4=deleted"""
        return self._param_int('status', 0, 4)

    def _param_log_kind(self):
        """log.kind  0-7"""
        return self._param_int('log_kind', 0, 7)

    def _param_link_kind(self):
        """link.kind  0-4"""
        return self._param_int('link_kind', 0, 4)

    def _param_dim(self):
        """measure.dim  0-4"""
        return self._param_int('dim', 0, 4)

    def _param_val(self):
        return self._param_float('val', 0, 100, 4)

    def _param_weight(self):
        return self._param_float('weight', 0.1, 0.9)

    def _param_priority(self):
        return self._param_float('priority', 0.0, 1.0)

    def _param_depth(self):
        """grp.depth  0=root  1=child"""
        return self._param_int('depth', 0, 1)
