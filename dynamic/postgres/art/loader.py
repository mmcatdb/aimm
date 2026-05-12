from typing_extensions import override
from core.loaders.postgres_loader import PostgresLoader, PostgresColumn, PostgresIndex

def export():
    return PostgresArtLoader()

class PostgresArtLoader(PostgresLoader):

    @override
    def _get_kinds(self):
        return _get_art_kinds()

    @override
    def _get_constraints(self):
        return [
            # node
            PostgresIndex('node', ['tag']),
            PostgresIndex('node', ['grp_id']),
            PostgresIndex('node', ['created_at']),
            PostgresIndex('node', ['is_active'], where='"is_active" = TRUE'),
            # link - PK is (src_id, dst_id); we need the reverse direction index
            PostgresIndex('link', ['dst_id']),
            PostgresIndex('link', ['src_id', 'kind']),
            # log
            PostgresIndex('log', ['node_id', 'occurred_at']),
            PostgresIndex('log', ['occurred_at']),
            PostgresIndex('log', ['kind']),
            # measure
            PostgresIndex('measure', ['node_id', 'dim', 'recorded_at']),
            PostgresIndex('measure', ['dim', 'recorded_at']),
            # doc
            PostgresIndex('doc', ['node_id']),
        ]

def _get_art_kinds() -> dict[str, list[PostgresColumn]]:
    grp = [
        PostgresColumn('grp_id',    'INTEGER',          primary_key=True),
        PostgresColumn('name',      'TEXT NOT NULL'),
        PostgresColumn('parent_id', 'INTEGER',          references='grp(grp_id)'),
        PostgresColumn('depth',     'INTEGER NOT NULL'),
        PostgresColumn('priority',  'FLOAT8 NOT NULL'),
    ]

    node = [
        PostgresColumn('node_id',    'INTEGER',              primary_key=True),
        PostgresColumn('tag',        'TEXT NOT NULL'),
        PostgresColumn('val_int',    'INTEGER NOT NULL'),
        PostgresColumn('val_float',  'FLOAT8 NOT NULL'),
        PostgresColumn('note',       'TEXT NOT NULL'),
        PostgresColumn('status',     'SMALLINT NOT NULL'),
        PostgresColumn('grp_id',     'INTEGER NOT NULL',     references='grp(grp_id)'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
        PostgresColumn('is_active',  'BOOLEAN NOT NULL'),
    ]

    link = [
        PostgresColumn('src_id',     'INTEGER NOT NULL',     primary_key=True, references='node(node_id)'),
        PostgresColumn('dst_id',     'INTEGER NOT NULL',     primary_key=True, references='node(node_id)'),
        PostgresColumn('kind',       'SMALLINT NOT NULL'),
        PostgresColumn('weight',     'FLOAT8 NOT NULL'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    log = [
        PostgresColumn('log_id',      'BIGINT',               primary_key=True),
        PostgresColumn('node_id',     'INTEGER NOT NULL',     references='node(node_id)'),
        PostgresColumn('kind',        'SMALLINT NOT NULL'),
        PostgresColumn('val',         'FLOAT8'),
        PostgresColumn('occurred_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    measure = [
        PostgresColumn('measure_id',  'BIGINT',               primary_key=True),
        PostgresColumn('node_id',     'INTEGER NOT NULL',     references='node(node_id)'),
        PostgresColumn('dim',         'SMALLINT NOT NULL'),
        PostgresColumn('val',         'FLOAT8 NOT NULL'),
        PostgresColumn('recorded_at', 'DATE NOT NULL'),
    ]

    doc = [
        PostgresColumn('doc_id',     'INTEGER',              primary_key=True),
        PostgresColumn('node_id',    'INTEGER NOT NULL',     references='node(node_id)'),
        PostgresColumn('body',       'TEXT NOT NULL'),
        PostgresColumn('meta',       'JSONB NOT NULL'),
        PostgresColumn('created_at', 'TIMESTAMPTZ NOT NULL'),
    ]

    return {
        'grp':     grp,
        'node':    node,
        'link':    link,
        'log':     log,
        'measure': measure,
        'doc':     doc,
    }
