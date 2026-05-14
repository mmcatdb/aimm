from typing_extensions import override
from core.loaders.mongo_loader import MongoLoader, MongoPostgresBuilder, MongoIndex
from core.query import parse_schema_id
from ...common.art.data_generator import ArtDataGenerator
from ...postgres.art.loader import _get_art_kinds

def export():
    return MongoArtLoader()

class MongoArtLoader(MongoLoader):
    # Collections created
    # ===================
    # grp        - flat group documents
    # node       - rich node documents; embeds grp, doc[], logs[], measures[], out_links[]
    # log        - flat log events
    # measure    - flat measurements
    # link       - flat directed edges
    # doc        - flat document records
    # --- JSON (MongoDB-specific) ---
    # node_rich  - deeply nested node documents (4-5 levels)
    # grp_tree   - group hierarchy with materialized path + ancestor/child lists
    # event_log  - polymorphic event journal (payload shape varies by event_type)
    # bucket     - time-series bucket pattern with pre-computed stats

    @override
    def _get_csv_kinds(self):
        b = MongoPostgresBuilder.create(_get_art_kinds())

        grp = b.plain_copy('grp')

        # Rich node document: embeds its group, all logs, measures, linked doc(s) and outgoing links.
        # doc / logs / measures / out_links are arrays so missing entries become [].
        node = b.document('node', {
            'node_id':    'node_id',
            'tag':        'tag',
            'val_int':    'val_int',
            'val_float':  'val_float',
            'note':       'note',
            'status':     'status',
            'grp_id':     'grp_id',
            'created_at': 'created_at',
            'is_active':  'is_active',
            # embedded group (1:1 - every node has a grp_id)
            'grp': b.nested('grp', {
                'grp_id':    'grp_id',
                'name':      'name',
                'depth':     'depth',
                'priority':  'priority',
                'parent_id': 'parent_id',
            }, parent_join='grp_id', child_join='grp_id'),
            # embedded document(s) - at most one per node
            'doc': b.nested('doc', {
                'doc_id':     'doc_id',
                'body':       'body',
                'meta':       'meta',
                'created_at': 'created_at',
            }, parent_join='node_id', child_join='node_id', is_array=True),
            # embedded log events
            'logs': b.nested('log', {
                'log_id':      'log_id',
                'kind':        'kind',
                'val':         'val',
                'occurred_at': 'occurred_at',
            }, parent_join='node_id', child_join='node_id', is_array=True),
            # embedded measurements
            'measures': b.nested('measure', {
                'measure_id':  'measure_id',
                'dim':         'dim',
                'val':         'val',
                'recorded_at': 'recorded_at',
            }, parent_join='node_id', child_join='node_id', is_array=True),
            # outgoing links (src_id == node_id)
            'out_links': b.nested('link', {
                'dst_id':     'dst_id',
                'kind':       'kind',
                'weight':     'weight',
                'created_at': 'created_at',
            }, parent_join='node_id', child_join='src_id', is_array=True),
        })

        log     = b.plain_copy('log')
        measure = b.plain_copy('measure')
        link    = b.plain_copy('link')
        doc     = b.plain_copy('doc')

        return [grp, node, log, measure, link, doc]

    def _allow_large_kinds(self) -> bool:
        _, scale = parse_schema_id(self._schema_id)
        return ArtDataGenerator.allow_large_kinds(scale)

    @override
    def _get_json_kinds(self):
        return ['node_rich', 'grp_tree', 'event_log', 'bucket'] if self._allow_large_kinds() else []

    @override
    def _get_constraints(self):
        return [
            # grp
            MongoIndex('grp', ['grp_id'], is_unique=True),
            MongoIndex('grp', ['depth']),
            MongoIndex('grp', ['parent_id']),
            # node (flat + nested path indexes)
            MongoIndex('node', ['node_id'], is_unique=True),
            MongoIndex('node', ['tag']),
            MongoIndex('node', ['status']),
            MongoIndex('node', ['grp_id']),
            MongoIndex('node', ['created_at']),
            MongoIndex('node', ['is_active']),
            MongoIndex('node', ['val_int']),
            MongoIndex('node', ['grp.depth']),
            MongoIndex('node', ['logs.kind']),
            MongoIndex('node', ['logs.occurred_at']),
            MongoIndex('node', ['measures.dim']),
            MongoIndex('node', ['measures.recorded_at']),
            MongoIndex('node', ['out_links.dst_id']),
            MongoIndex('node', ['out_links.kind']),
            # log
            MongoIndex('log', ['log_id'], is_unique=True),
            MongoIndex('log', ['node_id']),
            MongoIndex('log', ['kind']),
            MongoIndex('log', ['occurred_at']),
            MongoIndex('log', ['node_id', 'occurred_at']),
            # measure
            MongoIndex('measure', ['measure_id'], is_unique=True),
            MongoIndex('measure', ['node_id']),
            MongoIndex('measure', ['dim']),
            MongoIndex('measure', ['recorded_at']),
            MongoIndex('measure', ['node_id', 'dim', 'recorded_at']),
            MongoIndex('measure', ['dim', 'recorded_at']),
            # link
            MongoIndex('link', ['src_id', 'dst_id'], is_unique=True),
            MongoIndex('link', ['dst_id']),
            MongoIndex('link', ['kind']),
            MongoIndex('link', ['src_id', 'kind']),
            MongoIndex('link', ['weight']),
            # doc
            MongoIndex('doc', ['doc_id'], is_unique=True),
            MongoIndex('doc', ['node_id'], is_unique=True),
        ] + (
            # ---- JSON collections ----
            self.__get_json_constraints() if self._allow_large_kinds() else []
        )

    def __get_json_constraints(self):
        return [
            # node_rich
            MongoIndex('node_rich', ['node_id'], is_unique=True),
            MongoIndex('node_rich', ['identity.status']),
            MongoIndex('node_rich', ['identity.is_active']),
            MongoIndex('node_rich', ['identity.labels']),
            MongoIndex('node_rich', ['identity.flags.is_hot']),
            MongoIndex('node_rich', ['scores.tier']),
            MongoIndex('node_rich', ['scores.composite']),
            MongoIndex('node_rich', ['context.grp_id']),
            MongoIndex('node_rich', ['context.grp.depth']),
            MongoIndex('node_rich', ['connections.out_degree']),
            # grp_tree
            MongoIndex('grp_tree', ['grp_id'], is_unique=True),
            MongoIndex('grp_tree', ['depth']),
            MongoIndex('grp_tree', ['path']),
            MongoIndex('grp_tree', ['parent_id']),
            MongoIndex('grp_tree', ['stats.node_count']),
            # event_log
            MongoIndex('event_log', ['event_id'], is_unique=True),
            MongoIndex('event_log', ['node_id']),
            MongoIndex('event_log', ['event_type']),
            MongoIndex('event_log', ['occurred_at']),
            MongoIndex('event_log', ['actor.role']),
            MongoIndex('event_log', ['node_id', 'event_type']),
            MongoIndex('event_log', ['node_id', 'occurred_at']),
            # bucket
            MongoIndex('bucket', ['bucket_id'], is_unique=True),
            MongoIndex('bucket', ['node_id']),
            MongoIndex('bucket', ['dim']),
            MongoIndex('bucket', ['hour_start']),
            MongoIndex('bucket', ['node_id', 'dim']),
            MongoIndex('bucket', ['node_id', 'dim', 'hour_start']),
            MongoIndex('bucket', ['avg']),
            MongoIndex('bucket', ['stddev']),
        ]
