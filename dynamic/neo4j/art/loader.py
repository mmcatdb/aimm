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
        # Order determines which .tbl files must exist; must match data_generator output.
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
                grp_id:   toInteger(row[0]),
                name:     row[1],
                depth:    toInteger(row[3]),
                priority: toFloat(row[4])
            })
        ''')

        # parent_id may be empty for root groups
        self._load_csv('CHILD_OF', 'grp', '''
            WITH row
            WHERE row[2] <> ''
            MATCH (child:Grp  {grp_id: toInteger(row[0])}),
                  (parent:Grp {grp_id: toInteger(row[2])})
            CREATE (child)-[:CHILD_OF]->(parent)
        ''', 'creating Grp CHILD_OF relationships')

        self._load_csv('Node', 'node', '''
            CREATE (:Node {
                node_id:    toInteger(row[0]),
                tag:        row[1],
                val_int:    toInteger(row[2]),
                val_float:  toFloat(row[3]),
                note:       row[4],
                status:     toInteger(row[5]),
                grp_id:     toInteger(row[6]),
                created_at: datetime(row[7]),
                is_active:  (row[8] = 'true')
            })
        ''')

        self._create_relationship('BELONGS_TO', 'Node', 'grp_id', 'Grp', 'grp_id')

        self._load_csv('LINKED', 'link', '''
            MATCH (src:Node {node_id: toInteger(row[0])}),
                  (dst:Node {node_id: toInteger(row[1])})
            CREATE (src)-[:LINKED {
                kind:       toInteger(row[2]),
                weight:     toFloat(row[3]),
                created_at: datetime(row[4])
            }]->(dst)
        ''')

        self._load_csv('Log', 'log', '''
            CREATE (:Log {
                log_id:      toInteger(row[0]),
                node_id:     toInteger(row[1]),
                kind:        toInteger(row[2]),
                val:         CASE row[3] WHEN '' THEN null ELSE toFloat(row[3]) END,
                occurred_at: datetime(row[4])
            })
        ''')

        self._create_relationship('EVENT_OF', 'Log', 'node_id', 'Node', 'node_id')

        self._load_csv('Measure', 'measure', '''
            CREATE (:Measure {
                measure_id:  toInteger(row[0]),
                node_id:     toInteger(row[1]),
                dim:         toInteger(row[2]),
                val:         toFloat(row[3]),
                recorded_at: date(row[4])
            })
        ''')

        self._create_relationship('METRIC_OF', 'Measure', 'node_id', 'Node', 'node_id')

        self._load_csv('Doc', 'doc', '''
            CREATE (:Doc {
                doc_id:     toInteger(row[0]),
                node_id:    toInteger(row[1]),
                body:       row[2],
                meta:       row[3],
                created_at: datetime(row[4])
            })
        ''')

        self._create_relationship('DOC_OF', 'Doc', 'node_id', 'Node', 'node_id')
