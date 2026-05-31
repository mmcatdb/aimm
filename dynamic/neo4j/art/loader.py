from typing_extensions import override
from core.loaders.neo4j_loader import Neo4jLoader

def export():
    return Neo4jArtLoader()

class Neo4jArtLoader(Neo4jLoader):
    # Graph model
    # ===========
    # Nodes (labels):
    #     Grp     - small hierarchical groups
    #     Node    - main entities
    #     Log     - time-series event records
    #     Measure - periodic metric measurements
    #     Doc     - large document attached to a Node

    # Relationships:
    #     (Grp)-[:CHILD_OF]->(Grp)           parent/child hierarchy
    #     (Node)-[:BELONGS_TO]->(Grp)         node membership
    #     (Node)-[:LINKED]->(Node)            directed edge (replaces link table)
    #     (Log)-[:EVENT_OF]->(Node)           log owned by node
    #     (Measure)-[:METRIC_OF]->(Node)      measure owned by node
    #     (Doc)-[:DOC_OF]->(Node)             document owned by node

    # All properties mirror the PostgreSQL column names so queries stay comparable.

    @override
    def _get_kinds(self):
        # Order determines which .csv files must exist; must match data_generator output.
        return ['grp', 'node', 'link', 'log', 'measure', 'doc']

    @override
    def _get_constraints(self):
        return [
            # Unique node IDs
            'CREATE CONSTRAINT art_grp_id     IF NOT EXISTS FOR (n:Grp)     REQUIRE n.grp_id     IS UNIQUE',
            'CREATE CONSTRAINT art_node_id    IF NOT EXISTS FOR (n:Node)    REQUIRE n.node_id    IS UNIQUE',
            'CREATE CONSTRAINT art_log_id     IF NOT EXISTS FOR (n:Log)     REQUIRE n.log_id     IS UNIQUE',
            'CREATE CONSTRAINT art_measure_id IF NOT EXISTS FOR (n:Measure) REQUIRE n.measure_id IS UNIQUE',
            'CREATE CONSTRAINT art_doc_id     IF NOT EXISTS FOR (n:Doc)     REQUIRE n.doc_id     IS UNIQUE',
            # Lookup indexes
            'CREATE INDEX art_node_tag        IF NOT EXISTS FOR (n:Node)    ON (n.tag)',
            'CREATE INDEX art_node_status     IF NOT EXISTS FOR (n:Node)    ON (n.status)',
            'CREATE INDEX art_node_grp_id     IF NOT EXISTS FOR (n:Node)    ON (n.grp_id)',
            'CREATE INDEX art_node_created_at IF NOT EXISTS FOR (n:Node)    ON (n.created_at)',
            'CREATE INDEX art_log_node_id     IF NOT EXISTS FOR (n:Log)     ON (n.node_id)',
            'CREATE INDEX art_log_occurred_at IF NOT EXISTS FOR (n:Log)     ON (n.occurred_at)',
            'CREATE INDEX art_log_kind        IF NOT EXISTS FOR (n:Log)     ON (n.kind)',
            'CREATE INDEX art_measure_node_id IF NOT EXISTS FOR (n:Measure) ON (n.node_id)',
            'CREATE INDEX art_measure_dim     IF NOT EXISTS FOR (n:Measure) ON (n.dim)',
            'CREATE INDEX art_measure_rec_at  IF NOT EXISTS FOR (n:Measure) ON (n.recorded_at)',
            'CREATE INDEX art_doc_node_id     IF NOT EXISTS FOR (n:Doc)     ON (n.node_id)',
        ]

    @override
    def _load_data(self):
        self._load_csv('Grp', 'grp', '''
            CREATE (:Grp {
                grp_id:   toInteger(row.grp_id),
                name:     row.name,
                depth:    toInteger(row.depth),
                priority: toFloat(row.priority)
            })
        ''')

        # parent_id may be empty for root groups
        self._load_csv('CHILD_OF', 'grp', '''
            WITH row
            WHERE row.parent_id <> ''
            MATCH (child:Grp  {grp_id: toInteger(row.grp_id)}),
                  (parent:Grp {grp_id: toInteger(row.parent_id)})
            CREATE (child)-[:CHILD_OF]->(parent)
        ''', 'creating Grp CHILD_OF relationships')

        self._load_csv('Node', 'node', '''
            CREATE (:Node {
                node_id:    toInteger(row.node_id),
                tag:        row.tag,
                val_int:    toInteger(row.val_int),
                val_float:  toFloat(row.val_float),
                note:       row.note,
                status:     toInteger(row.status),
                grp_id:     toInteger(row.grp_id),
                created_at: datetime(row.created_at),
                is_active:  (row.is_active = 'true')
            })
        ''')

        self._create_relationship('BELONGS_TO', 'Node', 'grp_id', 'Grp', 'grp_id')

        self._load_csv('LINKED', 'link', '''
            MATCH (src:Node {node_id: toInteger(row.src_id)}),
                  (dst:Node {node_id: toInteger(row.dst_id)})
            CREATE (src)-[:LINKED {
                kind:       toInteger(row.kind),
                weight:     toFloat(row.weight),
                created_at: datetime(row.created_at)
            }]->(dst)
        ''')

        self._load_csv('Log', 'log', '''
            CREATE (:Log {
                log_id:      toInteger(row.log_id),
                node_id:     toInteger(row.node_id),
                kind:        toInteger(row.kind),
                val:         CASE row.val WHEN '' THEN null ELSE toFloat(row.val) END,
                occurred_at: datetime(row.occurred_at)
            })
        ''')

        self._create_relationship('EVENT_OF', 'Log', 'node_id', 'Node', 'node_id')

        self._load_csv('Measure', 'measure', '''
            CREATE (:Measure {
                measure_id:  toInteger(row.measure_id),
                node_id:     toInteger(row.node_id),
                dim:         toInteger(row.dim),
                val:         toFloat(row.val),
                recorded_at: date(row.recorded_at)
            })
        ''')

        self._create_relationship('METRIC_OF', 'Measure', 'node_id', 'Node', 'node_id')

        self._load_csv('Doc', 'doc', '''
            CREATE (:Doc {
                doc_id:     toInteger(row.doc_id),
                node_id:    toInteger(row.node_id),
                body:       row.body,
                meta:       row.meta,
                created_at: datetime(row.created_at)
            })
        ''')

        self._create_relationship('DOC_OF', 'Doc', 'node_id', 'Node', 'node_id')
