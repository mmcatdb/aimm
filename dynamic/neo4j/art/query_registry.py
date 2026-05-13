from typing_extensions import override
from core.drivers import DriverType
from ...common.art.query_registry import ArtQueryRegistry

def export():
    return Neo4jArtQueryRegistry()

class Neo4jArtQueryRegistry(ArtQueryRegistry[str]):

    def __init__(self):
        super().__init__(DriverType.NEO4J)

    @override
    def _register_queries(self):
        self._register_basic_queries()
        self._register_extended_queries()
        self._register_write_queries()

    def _register_basic_queries(self):
        # Coverage map
        # ============
        # pk-0      ... pk-1         Node-by-ID lookups (PK index seek)
        # scan-0    ... scan-20      Single-label property filters
        # agg-0     ... agg-6        Aggregation (count, sum, avg, collect, percentile)
        # join-0    ... join-13      Multi-hop MATCH traversals (BELONGS_TO, EVENT_OF, etc.)
        # optional-0... optional-5   OPTIONAL MATCH patterns
        # path-0    ... path-8       Variable-length paths and shortestPath
        # subq-0    ... subq-7       CALL { } subqueries and EXISTS {}
        # collect-0 ... collect-5    COLLECT / list comprehension / UNWIND roundtrips
        # with-0    ... with-8       Multi-stage WITH pipelines
        # order-0   ... order-5      ORDER BY / SKIP / LIMIT
        # cond-0    ... cond-5       CASE / coalesce / conditional logic
        # pattern-0 ... pattern-8    Pattern comprehensions and predicate functions
        # set-op-0  ... set-op-4     UNION / UNION ALL
        # window-0  ... window-5     apoc-free window-style with COLLECT + UNWIND
        # graph-0   ... graph-9      Graph algorithms expressed in Cypher
        #                         (degree, triangles, clustering, hub analysis)
        # complex-0 ... complex-9    Multi-feature compositions

        #region Point lookups

        self._query('pk-0', 'Index seek on Node by node_id (PK)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            RETURN n.node_id, n.tag, n.val_int, n.val_float, n.status, n.grp_id, n.created_at, n.is_active
        """)

        self._query('pk-1', 'Index seek on Node by node_id + follow BELONGS_TO', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:BELONGS_TO]->(g:Grp)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            RETURN n.node_id, n.tag, n.note, g.name AS grp_name, d.body AS doc_body
        """)

        #endregion
        #region Single-label property filters

        self._query('scan-0', 'Label scan Node filtered by tag (index)', lambda s: f"""
            MATCH (n:Node {{tag: '{s._param_tag()}'}})
            RETURN n.node_id, n.tag, n.val_int, n.grp_id
        """)

        self._query('scan-1', 'Label scan Node filtered by val_int exact', lambda s: f"""
            MATCH (n:Node)
            WHERE n.val_int = {s._param_val_int()}
            RETURN n.node_id, n.tag, n.val_int
        """)

        self._query('scan-2', 'Label scan Node filtered by val_int range', lambda s: f"""
            MATCH (n:Node)
            WHERE n.val_int >= {s._param_int('val_lo', 1, 800)}
              AND n.val_int <= {s._param_int('val_hi', 200, 1000)}
            RETURN n.node_id, n.tag, n.val_int
        """)

        self._query('scan-3', 'Label scan Node filtered by created_at >= date', lambda s: f"""
            MATCH (n:Node)
            WHERE n.created_at >= datetime('{s._param_date_minus_days(30, 365)}')
            RETURN n.node_id, n.tag, n.created_at
        """)

        self._query('scan-4', 'Label scan Node filtered by created_at BETWEEN two dates', lambda s: f"""
            MATCH (n:Node)
            WHERE n.created_at >= datetime('{s._param_date_minus_days(60, 180)}')
              AND n.created_at <  datetime('{s._param_date_minus_days(1,   59)}')
            RETURN n.node_id, n.tag, n.created_at
        """)

        self._query('scan-5', 'Label scan Node filtered by status', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status = {s._param_status()}
            RETURN n.node_id, n.tag, n.status
        """)

        self._query('scan-6', 'Composite filter: is_active=true + grp_id (index on both)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
              AND n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, n.grp_id
        """)

        self._query('scan-7', 'Composite AND: tag (indexed) + val_int (non-indexed)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.tag = '{s._param_tag()}'
              AND n.val_int > {s._param_val_int()}
            RETURN n.node_id, n.tag, n.val_int, n.grp_id
        """)

        self._query('scan-8', 'Composite AND: val_int range + status', lambda s: f"""
            MATCH (n:Node)
            WHERE n.val_int > {s._param_int('val_lo', 400, 800)}
              AND n.status = {s._param_status()}
            RETURN n.node_id, n.val_int, n.val_float, n.status
        """)

        self._query('scan-9', 'Composite OR: val_int low OR status match', lambda s: f"""
            MATCH (n:Node)
            WHERE n.val_int < {s._param_int('val_threshold', 100, 300)}
               OR n.status = {s._param_status()}
            RETURN n.node_id, n.tag, n.val_int, n.status
        """)

        self._query('scan-10', 'STARTS WITH on note (non-indexed text prefix scan)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.note STARTS WITH '{s._param('prefix', lambda: chr(ord('a') + s._rng_int(0, 25)))}'
            RETURN n.node_id, n.note
            LIMIT 200
        """)

        self._query('scan-11', 'Log nodes where val IS NULL', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val IS NULL
              AND l.kind = {s._param_log_kind()}
            RETURN l.log_id, l.node_id, l.kind, l.occurred_at
        """)

        self._query('scan-12', 'Log nodes where val IS NOT NULL for a given node', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val IS NOT NULL
              AND l.node_id = {s._param_node_id()}
            RETURN l.log_id, l.val, l.occurred_at
        """)

        self._query('scan-13', 'Node IN small list of grp_ids (5 values)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id IN [{s._param_grp_ids(5, 5)}]
            RETURN n.node_id, n.tag, n.grp_id
        """)

        self._query('scan-14', 'Node IN large list of grp_ids (50 values)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id IN [{s._param_grp_ids(50, 50)}]
            RETURN n.node_id, n.tag, n.grp_id
        """)

        self._query('scan-15', 'Node NOT IN list of grp_ids + is_active filter', lambda s: f"""
            MATCH (n:Node)
            WHERE NOT n.grp_id IN [{s._param_grp_ids(10, 20)}]
              AND n.is_active = true
            RETURN n.node_id, n.tag, n.grp_id
        """)

        self._query('scan-16', 'ORDER BY indexed created_at DESC + LIMIT', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            RETURN n.node_id, n.tag, n.created_at
            ORDER BY n.created_at DESC
            LIMIT {s._param_limit(10, 4)}
        """)

        self._query('scan-17', 'ORDER BY non-indexed val_float DESC + LIMIT', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status = {s._param_status()}
            RETURN n.node_id, n.val_int, n.val_float
            ORDER BY n.val_float DESC
            LIMIT {s._param_limit(10, 4)}
        """)

        self._query('scan-18', 'DISTINCT status values in a group', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN DISTINCT n.status
            ORDER BY n.status
        """)

        self._query('scan-19', 'COUNT(*) with tag + is_active filter', lambda s: f"""
            MATCH (n:Node)
            WHERE n.tag = '{s._param_tag()}'
              AND n.is_active = true
            RETURN count(n) AS cnt
        """)

        self._query('scan-20', 'COUNT(DISTINCT grp_id) per status', lambda s: f"""
            MATCH (n:Node)
            WHERE n.created_at >= datetime('{s._param_date_minus_days(30, 365)}')
            RETURN n.status, count(DISTINCT n.grp_id) AS distinct_grps
            ORDER BY n.status
        """)

        #endregion
        #region Aggregation

        self._query('agg-0', 'SUM/AVG/MIN/MAX of log.val per node', lambda s: f"""
            MATCH (l:Log)
            WHERE l.node_id = {s._param_node_id()}
              AND l.val IS NOT NULL
            RETURN count(l) AS n,
                   sum(l.val) AS total,
                   avg(l.val) AS avg_val,
                   min(l.val) AS min_val,
                   max(l.val) AS max_val
        """)

        self._query('agg-1', 'GROUP BY status: count + avg val_float', lambda s: f"""
            MATCH (n:Node)
            RETURN n.status AS status, count(n) AS cnt, avg(n.val_float) AS avg_f
            ORDER BY status
        """)

        self._query('agg-2', 'GROUP BY grp_id: count + sum val_int (active only)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            RETURN n.grp_id AS grp_id, count(n) AS cnt, sum(n.val_int) AS sum_val
            ORDER BY grp_id
        """)

        self._query('agg-3', 'GROUP BY grp_id + status: count', lambda s: f"""
            MATCH (n:Node)
            RETURN n.grp_id AS grp_id, n.status AS status, count(n) AS cnt
            ORDER BY grp_id, status
        """)

        self._query('agg-4', 'GROUP BY grp_id HAVING count >= threshold', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            WITH n.grp_id AS grp_id, count(n) AS cnt, sum(n.val_int) AS total
            WHERE cnt >= {s._param_int('min_cnt', 20, 100)}
            RETURN grp_id, cnt, total
            ORDER BY total DESC
        """)

        self._query('agg-5', 'Top-N groups by avg val_float', lambda s: f"""
            MATCH (n:Node)
            RETURN n.grp_id AS grp_id, count(n) AS cnt, avg(n.val_float) AS avg_f
            ORDER BY avg_f DESC
            LIMIT {s._param_limit(10, 3)}
        """)

        self._query('agg-6', 'Conditional count with CASE inside aggregation', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN
                count(CASE WHEN n.is_active THEN 1 END) AS active_cnt,
                count(CASE WHEN NOT n.is_active THEN 1 END) AS inactive_cnt,
                avg(n.val_int) AS avg_val,
                percentileCont(n.val_float, 0.5) AS median_f
        """)

        #endregion
        #region Multi-hop MATCH traversals

        self._query('join-0', 'Node -> Grp (BELONGS_TO, N:1)', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, g.name AS grp_name, g.depth
        """)

        self._query('join-1', 'Node <- Log (EVENT_OF, 1:N) most recent events', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            RETURN l.log_id, l.kind, l.val, l.occurred_at
            ORDER BY l.occurred_at DESC
            LIMIT 20
        """)

        self._query('join-2', 'Node -> Grp -> parent Grp (2-hop hierarchy)', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)-[:CHILD_OF]->(parent:Grp)
            WHERE n.is_active = true
            RETURN n.node_id, n.tag, g.name AS grp_name, parent.name AS parent_name
            LIMIT {s._param_limit(20, 3)}
        """)

        self._query('join-3', 'Node + Log + Measure (3-entity pattern)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)<-[:METRIC_OF]-(m:Measure)
            WHERE n.node_id = {s._param_node_id()}
              AND l.val IS NOT NULL
            RETURN n.node_id, l.log_id, l.kind, l.val, m.measure_id, m.dim, m.val AS measure_val
            LIMIT 50
        """)

        self._query('join-4', 'Node + Grp + Log aggregate (join + agg)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.grp_id = {s._param_grp_id()}
              AND l.val IS NOT NULL
            RETURN n.node_id, n.tag, g.name, count(l) AS log_cnt, avg(l.val) AS avg_val
        """)

        self._query('join-5', 'Node + Doc (1:1 optional attachment)', lambda s: f"""
            MATCH (d:Doc)-[:DOC_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, d.doc_id, d.created_at
            LIMIT 50
        """)

        self._query('join-6', '4-entity: Node + Grp + Log + Measure', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            MATCH (n)<-[:METRIC_OF]-(m:Measure)
            WHERE n.status = {s._param_status()}
              AND m.dim = {s._param_dim()}
            RETURN n.node_id, g.name AS grp, l.kind, l.val, m.val AS measure_val
            LIMIT 100
        """)

        self._query('join-7', 'Node LINKED to Node (1-hop out-neighbors)', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(dst:Node)
            RETURN dst.node_id, dst.tag, r.kind, r.weight
            ORDER BY r.weight DESC
        """)

        self._query('join-8', 'LINKED + BELONGS_TO (neighbor grp lookup)', lambda s: f"""
            MATCH (src:Node)-[r:LINKED]->(dst:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE src.node_id = {s._param_node_id()}
            RETURN dst.node_id, dst.tag, g.name AS dst_grp, r.weight
        """)

        self._query('join-9', 'Anti-join: nodes with no Doc', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND NOT EXISTS {{ MATCH (d:Doc)-[:DOC_OF]->(n) }}
            RETURN n.node_id, n.tag, n.status
        """)

        self._query('join-10', 'Grp -> children Grps -> Nodes (3-hop)', lambda s: f"""
            MATCH (parent:Grp {{grp_id: {s._param_grp_id()}}})<-[:CHILD_OF]-(child:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(child)
            RETURN parent.name AS parent, child.name AS child_grp, n.node_id, n.tag
            LIMIT {s._param_limit(50, 2)}
        """)

        self._query('join-11', 'Star: Node + Grp + Doc + Log agg subquery', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            MATCH (d:Doc)-[:DOC_OF]->(n)
            WITH n, g, d
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val IS NOT NULL
                RETURN count(l) AS log_cnt, avg(l.val) AS avg_log_val
            }}
            WITH * WHERE n.is_active = true
            RETURN n.node_id, n.tag, g.name, d.doc_id, log_cnt, avg_log_val
            LIMIT 30
        """)

        self._query('join-12', 'In-neighbors via LINKED reverse direction', lambda s: f"""
            MATCH (src:Node)-[r:LINKED]->(dst:Node {{node_id: {s._param_node_id()}}})
            RETURN src.node_id, src.tag, r.kind, r.weight
            ORDER BY r.weight DESC
        """)

        self._query('join-13', 'Measure filtered by dim + date + node join', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE m.dim = {s._param_dim()}
              AND m.recorded_at >= date('{s._param_date_minus_days(30, 180)}')
            RETURN n.node_id, n.tag, g.name, m.dim, m.val, m.recorded_at
            LIMIT 100
        """)

        #endregion
        #region OPTIONAL MATCH

        self._query('optional-0', 'Node with OPTIONAL Doc (left join equivalent)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            RETURN n.node_id, n.tag, d.doc_id, d.created_at
        """)

        self._query('optional-1', 'Node with OPTIONAL Log count', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status = {s._param_status()}
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            RETURN n.node_id, n.tag, count(l) AS log_cnt
            ORDER BY log_cnt DESC
            LIMIT 50
        """)

        self._query('optional-2', 'Node with OPTIONAL Measure per dim', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n)
            WHERE m.dim = {s._param_dim()}
            RETURN n.node_id, n.tag, m.measure_id, m.val, m.recorded_at
        """)

        self._query('optional-3', 'Grp with OPTIONAL child Grps', lambda s: f"""
            MATCH (g:Grp)
            WHERE g.depth = 0
            OPTIONAL MATCH (child:Grp)-[:CHILD_OF]->(g)
            RETURN g.grp_id, g.name, child.grp_id AS child_id, child.name AS child_name
        """)

        self._query('optional-4', 'Node + OPTIONAL LINKED out-neighbors + OPTIONAL Doc', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            OPTIONAL MATCH (n)-[r:LINKED]->(nb:Node)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            RETURN n.node_id, n.tag, collect(nb.node_id) AS neighbors, d.doc_id
        """)

        self._query('optional-5', 'OPTIONAL double hop: node + optional 2-hop neighbor via LINKED', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND n.is_active = true
            OPTIONAL MATCH (n)-[:LINKED]->(mid:Node)-[:LINKED]->(far:Node)
            RETURN n.node_id, n.tag, count(DISTINCT mid) AS mid_cnt, count(DISTINCT far) AS far_cnt
            LIMIT 100
        """)

        #endregion
        #region Variable-length paths and shortestPath

        self._query('path-0', '1-3 hop path from node via LINKED', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..3]->(dst:Node)
            RETURN DISTINCT dst.node_id, dst.tag
            LIMIT {s._param_limit(50, 2)}
        """)

        self._query('path-1', 'shortestPath between two nodes via LINKED', lambda s: f"""
            MATCH (a:Node {{node_id: {s._param_node_id()}}}),
                  (b:Node {{node_id: {s._param_node_id('node_id_b')}}})
            MATCH p = shortestPath((a)-[:LINKED*..6]->(b))
            RETURN length(p) AS hops, [x IN nodes(p) | x.node_id] AS path_ids
        """)

        self._query('path-2', 'All paths <= 2 hops from node via LINKED (limited)', lambda s: f"""
            MATCH p = (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..2]->(dst:Node)
            RETURN DISTINCT dst.node_id, dst.tag, length(p) AS depth
            LIMIT 100
        """)

        self._query('path-3', 'Ancestor path via CHILD_OF (root search)', lambda s: f"""
            MATCH p = (g:Grp {{grp_id: {s._param_grp_id()}}})-[:CHILD_OF*0..5]->(root:Grp)
            WHERE NOT EXISTS {{ MATCH (root)-[:CHILD_OF]->(:Grp) }}
            RETURN [x IN nodes(p) | x.grp_id] AS ancestor_ids,
                   [x IN nodes(p) | x.name]   AS ancestor_names,
                   length(p) AS depth
        """)

        self._query('path-4', 'All descendants of root Grp via CHILD_OF reverse', lambda s: f"""
            MATCH p = (root:Grp {{depth: 0, grp_id: {s._param_grp_id()}}})<-[:CHILD_OF*1..5]-(desc:Grp)
            RETURN desc.grp_id, desc.name, length(p) AS depth
        """)

        self._query('path-5', 'Reachability: count nodes reachable in 1..4 hops via LINKED', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..4]->(dst:Node)
            RETURN count(DISTINCT dst) AS reachable_cnt
        """)

        self._query('path-6', 'allShortestPaths between two nodes', lambda s: f"""
            MATCH (a:Node {{node_id: {s._param_node_id()}}}),
                  (b:Node {{node_id: {s._param_node_id('node_id_b')}}})
            MATCH p = allShortestPaths((a)-[:LINKED*]->(b))
            RETURN length(p) AS hops, [x IN nodes(p) | x.node_id] AS path_ids
            LIMIT 10
        """)

        self._query('path-7', 'Common reachable nodes from two sources (intersection via LINKED)', lambda s: f"""
            MATCH (a:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..2]->(common:Node)
            MATCH (b:Node {{node_id: {s._param_node_id('node_id_b')}}})-[:LINKED*1..2]->(common)
            RETURN DISTINCT common.node_id, common.tag
            LIMIT 50
        """)

        self._query('path-8', 'Variable-length LINKED with weight filter on each hop', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[hops:LINKED*1..3]->(dst:Node)
            WHERE ALL(r IN hops WHERE r.weight > {s._param_weight()})
            RETURN DISTINCT dst.node_id, dst.tag, size(hops) AS depth
            LIMIT 100
        """)

        #endregion
        #region CALL {} subqueries

        self._query('subq-0', 'CALL subquery: count events per node in a group', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                RETURN count(l) AS log_cnt
            }}
            RETURN n.node_id, n.tag, log_cnt
            ORDER BY log_cnt DESC
            LIMIT 20
        """)

        self._query('subq-1', 'CALL subquery: latest log per node', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status = {s._param_status()}
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val IS NOT NULL
                RETURN l ORDER BY l.occurred_at DESC LIMIT 1
            }}
            RETURN n.node_id, n.tag, l.log_id, l.kind, l.val, l.occurred_at
            LIMIT 50
        """)

        self._query('subq-2', 'EXISTS {} anti-pattern: nodes with no outgoing LINKED', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND NOT EXISTS {{ MATCH (n)-[:LINKED]->(:Node) }}
            RETURN n.node_id, n.tag, n.status
        """)

        self._query('subq-3', 'EXISTS {} positive: nodes that have a Log with val > threshold', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
              AND EXISTS {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val > {s._param_float('val_threshold', 50.0, 90.0)}
              }}
            RETURN n.node_id, n.tag, n.grp_id
            LIMIT 100
        """)

        self._query('subq-4', 'CALL subquery: avg measure val per dim inside node loop', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND n.is_active = true
            CALL (n) {{
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN avg(m.val) AS avg_measure, count(m) AS measure_cnt
            }}
            RETURN n.node_id, n.tag, avg_measure, measure_cnt
        """)

        self._query('subq-5', 'Correlated EXISTS: nodes whose group has a root ancestor', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE EXISTS {{
                MATCH (g)-[:CHILD_OF*0..5]->(root:Grp)
                WHERE root.depth = 0 AND root.grp_id = {s._param_grp_id()}
            }}
            RETURN n.node_id, n.tag, g.grp_id, g.name
            LIMIT 100
        """)

        self._query('subq-6', 'CALL subquery: max measure val per node + outer filter', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status IN [0, 1]
            CALL (n) {{
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                WHERE m.dim = {s._param_dim()}
                RETURN max(m.val) AS max_val
            }}
            WITH * WHERE max_val > {s._param_int('threshold', 50, 500)}
            RETURN n.node_id, n.tag, max_val
            ORDER BY max_val DESC
            LIMIT 50
        """)

        self._query('subq-7', 'CALL UNION subquery: combine Log and Measure events for a node', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val IS NOT NULL
                RETURN 'log' AS kind, l.val AS val, l.occurred_at AS ts
              UNION ALL
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN 'measure' AS kind, m.val AS val, datetime({{epochSeconds: 0}}) AS ts
            }}
            RETURN kind, val, ts
            ORDER BY ts DESC
            LIMIT 50
        """)

        #endregion
        #region COLLECT / UNWIND / list operations

        self._query('collect-0', 'COLLECT neighbor node_ids into a list', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(nb:Node)
            RETURN n.node_id, collect(nb.node_id) AS neighbor_ids, count(nb) AS degree
        """)

        self._query('collect-1', 'COLLECT tags per group into list + list size', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.grp_id = {s._param_grp_id()}
            RETURN g.name, collect(DISTINCT n.tag) AS tags, size(collect(DISTINCT n.tag)) AS tag_cnt
        """)

        self._query('collect-2', 'UNWIND list of IDs for batch lookup', lambda s: f"""
            WITH [{s._param_node_ids(10, 20)}] AS ids
            UNWIND ids AS nid
            MATCH (n:Node {{node_id: nid}})
            RETURN n.node_id, n.tag, n.val_int, n.status
        """)

        self._query('collect-3', 'Collect Measure vals into list then compute via UNWIND', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE m.dim = {s._param_dim()}
            WITH collect(m.val) AS vals
            UNWIND vals AS v
            RETURN v
            ORDER BY v
        """)

        self._query('collect-4', 'List comprehension: filter neighbor tags', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(nb:Node)
            WITH n, collect(nb) AS neighbors
            RETURN n.node_id,
                   [x IN neighbors WHERE x.is_active | x.tag]  AS active_tags,
                   [x IN neighbors WHERE NOT x.is_active | x.node_id] AS inactive_ids
        """)

        self._query('collect-5', 'Collect Log kinds per node then size', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, collect(l.kind) AS kinds
            RETURN n.node_id, n.tag, size(kinds) AS event_cnt, kinds
            LIMIT 50
        """)

        #endregion
        #region Multi-stage WITH pipelines

        self._query('with-0', '2-stage: filter nodes -> aggregate logs', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND n.is_active = true
            WITH n
            MATCH (l:Log)-[:EVENT_OF]->(n)
            WHERE l.val IS NOT NULL
            RETURN n.node_id, n.tag, count(l) AS log_cnt, avg(l.val) AS avg_val
        """)

        self._query('with-1', '3-stage: group nodes -> rank by metric -> top-K', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WITH g, count(n) AS node_cnt, sum(n.val_int) AS total_val
            ORDER BY total_val DESC
            WITH collect({{g: g, node_cnt: node_cnt, total_val: total_val}}) AS ranked
            UNWIND range(0, size(ranked)-1) AS idx
            WITH ranked[idx] AS row, idx + 1 AS rnk
            WHERE rnk <= {s._param_int('top_k', 5, 20)}
            RETURN row.g.grp_id, row.g.name, row.node_cnt, row.total_val, rnk
        """)

        self._query('with-2', 'Compute per-node score then filter outliers', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            WITH n, (n.val_int * 1.0 / (n.val_float + 0.001)) AS score
            WHERE score > {s._param_int('min_score', 100, 5000)}
            RETURN n.node_id, n.tag, n.val_int, n.val_float, score
            ORDER BY score DESC
            LIMIT 50
        """)

        self._query('with-3', 'Per-group avg then filter nodes above group avg', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WITH g, avg(n.val_int) AS grp_avg
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WHERE n.val_int > grp_avg
            RETURN n.node_id, n.tag, n.val_int, grp_avg
        """)

        self._query('with-4', 'Log time-bucketing by day with WITH pipeline', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND l.occurred_at >= datetime('{s._param_date_minus_days(7, 90)}')
            WITH date(l.occurred_at) AS day, count(l) AS cnt, avg(l.val) AS avg_val
            RETURN day, cnt, avg_val
            ORDER BY day
        """)

        self._query('with-5', 'Degree calculation then select top-degree nodes', lambda s: f"""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:LINKED]->(out:Node)
            WITH n, count(out) AS out_degree
            OPTIONAL MATCH (in_n:Node)-[:LINKED]->(n)
            WITH n, out_degree, count(in_n) AS in_degree
            RETURN n.node_id, n.tag, out_degree, in_degree, out_degree + in_degree AS total_degree
            ORDER BY total_degree DESC
            LIMIT {s._param_limit(20, 3)}
        """)

        self._query('with-6', '4-stage: nodes -> logs -> rolling window -> outlier detect', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            MATCH (l:Log)-[:EVENT_OF]->(n)
            WHERE l.val IS NOT NULL
            WITH n, collect({{kind: l.kind, val: l.val, ts: l.occurred_at}}) AS events
            UNWIND events AS e
            WITH n, e, size(events) AS total_events, reduce(s=0.0, x IN events | s + x.val) / toFloat(size(events)) AS global_avg
            WHERE e.val > global_avg * 1.5
            RETURN n.node_id, e.kind AS kind, e.val AS val, e.ts AS ts, global_avg
        """)

        self._query('with-7', 'Collect per-group top-5 nodes then UNWIND for flat output', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 1
            WITH g, n ORDER BY n.val_int DESC
            WITH g, collect(n)[0..5] AS top_nodes
            UNWIND top_nodes AS tn
            RETURN g.grp_id, g.name, tn.node_id, tn.tag, tn.val_int
        """)

        self._query('with-8', 'Running measure total via UNWIND + reduce', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE m.dim = {s._param_dim()}
            WITH collect({{val: m.val, ts: m.recorded_at}}) AS measurements
            UNWIND measurements AS item
            WITH item, measurements, reduce(total = 0.0, x IN [x IN measurements WHERE x.ts <= item.ts] | total + x.val) AS running_total
            RETURN item.ts AS date, item.val AS val, running_total
            ORDER BY date
        """)

        #endregion
        #region ORDER BY / SKIP / LIMIT

        self._query('order-0', 'ORDER BY relationship property (weight DESC)', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(dst:Node)
            RETURN dst.node_id, dst.tag, r.kind, r.weight
            ORDER BY r.weight DESC
        """)

        self._query('order-1', 'ORDER BY computed expression', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, (n.val_int * n.val_float) AS score
            ORDER BY score DESC
            LIMIT {s._param_limit(20, 3)}
        """)

        self._query('order-2', 'SKIP + LIMIT for keyset-style pagination', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            RETURN n.node_id, n.tag, n.created_at
            ORDER BY n.created_at DESC
            SKIP  {s._param_skip(0, 500)}
            LIMIT {s._param_limit(10, 3)}
        """)

        self._query('order-3', 'Multi-column ORDER BY (status ASC, val_int DESC)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, n.status, n.val_int
            ORDER BY n.status ASC, n.val_int DESC
        """)

        self._query('order-4', 'ORDER BY aggregate result', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 0
            WITH g, count(l) AS log_cnt, avg(l.val) AS avg_val
            RETURN g.grp_id, g.name, log_cnt, avg_val
            ORDER BY log_cnt DESC, avg_val DESC
            LIMIT {s._param_limit(10, 3)}
        """)

        self._query('order-5', 'ORDER BY with NULLS handling (val nullable)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN l.log_id, l.val, l.occurred_at
            ORDER BY l.val DESC, l.occurred_at DESC
            LIMIT 100
        """)

        #endregion
        #region CASE / coalesce / conditional logic

        self._query('cond-0', 'CASE expression to bucket val_int', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag, n.val_int,
                   CASE
                       WHEN n.val_int < 250  THEN 'low'
                       WHEN n.val_int < 500  THEN 'medium'
                       WHEN n.val_int < 750  THEN 'high'
                       ELSE 'very_high'
                   END AS bucket
        """)

        self._query('cond-1', 'coalesce: fill missing log.val with 0', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.node_id = {s._param_node_id()}
            RETURN l.log_id, l.kind, coalesce(l.val, 0.0) AS safe_val, l.occurred_at
        """)

        self._query('cond-2', 'CASE in ORDER BY for custom sort priority', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            RETURN n.node_id, n.tag, n.status
            ORDER BY
                CASE n.status
                    WHEN 0 THEN 1
                    WHEN 2 THEN 2
                    ELSE 3
                END ASC,
                n.val_int DESC
            LIMIT 50
        """)

        self._query('cond-3', 'Conditional aggregate: count by CASE bucket', lambda s: f"""
            MATCH (n:Node)
            WITH
                count(CASE WHEN n.status = 0 THEN 1 END) AS active_cnt,
                count(CASE WHEN n.status = 1 THEN 1 END) AS inactive_cnt,
                count(CASE WHEN n.status >= 2 THEN 1 END) AS other_cnt,
                avg(CASE WHEN n.is_active THEN n.val_float END) AS active_avg_f
            RETURN active_cnt, inactive_cnt, other_cnt, active_avg_f
        """)

        self._query('cond-4', 'NULLIF to guard division by zero', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, count(l) AS log_cnt, sum(coalesce(l.val, 0.0)) AS log_sum
            RETURN n.node_id, n.tag, log_cnt, log_sum,
                   log_sum / toFloat(CASE log_cnt WHEN 0 THEN null ELSE log_cnt END) AS avg_safe
        """)

        self._query('cond-5', 'CASE with EXISTS subpattern inside RETURN', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN n.node_id, n.tag,
                   CASE WHEN EXISTS {{ (n)-[:LINKED]->(:Node) }} THEN 'has_out' ELSE 'isolated' END AS connectivity,
                   CASE WHEN EXISTS {{ (:Doc)-[:DOC_OF]->(n) }} THEN 'has_doc' ELSE 'no_doc' END AS doc_status
        """)

        #endregion
        #region Pattern comprehensions and predicate functions

        self._query('pattern-0', 'Pattern comprehension: collect neighbor node_ids inline', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            RETURN n.node_id, n.tag,
                   [(n)-[:LINKED]->(nb:Node) | nb.node_id] AS out_neighbor_ids,
                   size([(n)-[:LINKED]->(nb:Node) | nb.node_id]) AS out_degree
        """)

        self._query('pattern-1', 'Pattern comprehension with WHERE filter inside', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND n.is_active = true
            RETURN n.node_id, n.tag,
                   [(n)-[r:LINKED]->(nb:Node) WHERE r.weight > {s._param_weight()} | nb.tag] AS heavy_neighbors
            LIMIT 50
        """)

        self._query('pattern-2', 'ALL() predicate: all out-neighbors are active', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(nb:Node)
            WITH n, collect(nb) AS neighbors
            WHERE ALL(x IN neighbors WHERE x.is_active = true)
            RETURN n.node_id, n.tag, size(neighbors) AS deg
        """)

        self._query('pattern-3', 'ANY() predicate: any neighbor has high val_int', lambda s: f"""
            MATCH (n:Node)-[:LINKED]->(nb:Node)
            WITH n, collect(nb) AS neighbors
            WHERE ANY(x IN neighbors WHERE x.val_int > {s._param_int('hi', 800, 999)})
            RETURN n.node_id, n.tag, size(neighbors) AS deg
            LIMIT 100
        """)

        self._query('pattern-4', 'NONE() predicate: no neighbor has status=4 (deleted)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
              AND n.grp_id = {s._param_grp_id()}
            WITH n, [(n)-[:LINKED]->(nb:Node) | nb] AS neighbors
            WHERE NONE(x IN neighbors WHERE x.status = 4)
              AND size(neighbors) > 0
            RETURN n.node_id, n.tag, size(neighbors) AS deg
        """)

        self._query('pattern-5', 'SINGLE() predicate: exactly one neighbor matches', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            WITH n, [(n)-[:LINKED]->(nb:Node) | nb] AS neighbors
            WHERE SINGLE(x IN neighbors WHERE x.status = {s._param_status()})
            RETURN n.node_id, n.tag, size(neighbors) AS deg
        """)

        self._query('pattern-6', 'List comprehension: map + filter measure vals', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            WITH n, [(m:Measure)-[:METRIC_OF]->(n) WHERE m.dim = {s._param_dim()} | m.val] AS vals
            RETURN n.node_id,
                   vals,
                   reduce(s = 0.0, v IN vals | s + v) AS total,
                   size([v IN vals WHERE v > {s._param_int('thr', 50, 300)}]) AS above_thr_cnt
        """)

        self._query('pattern-7', 'Pattern comprehension in WITH then UNWIND', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WITH n, [(l:Log)-[:EVENT_OF]->(n) WHERE l.val IS NOT NULL | l.val] AS log_vals
            WHERE size(log_vals) > 0
            WITH n, log_vals, reduce(s=0.0, v IN log_vals | s+v) / toFloat(size(log_vals)) AS avg_val
            RETURN n.node_id, n.tag, size(log_vals) AS cnt, avg_val
            ORDER BY avg_val DESC
            LIMIT 20
        """)

        self._query('pattern-8', 'nodes() + relationships() on a path', lambda s: f"""
            MATCH p = (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..3]->(dst:Node)
            WITH p, length(p) AS hops
            ORDER BY hops
            LIMIT 5
            RETURN [x IN nodes(p) | x.node_id] AS path_ids,
                   [r IN relationships(p) | r.weight] AS edge_weights,
                   reduce(w = 1.0, r IN relationships(p) | w * r.weight) AS path_weight
        """)

        #endregion
        #region UNION / UNION ALL

        self._query('set-op-0', 'UNION ALL: two disjoint status filters on Node', lambda s: f"""
            MATCH (n:Node) WHERE n.status = 0 AND n.grp_id = {s._param_grp_id()}
                RETURN n.node_id AS id, n.tag AS tag, 'active' AS src
            UNION ALL
            MATCH (n:Node) WHERE n.status = 1 AND n.grp_id = {s._param_grp_id()}
                RETURN n.node_id AS id, n.tag AS tag, 'inactive' AS src
        """)

        self._query('set-op-1', 'UNION (distinct): nodes reachable from two grps via BELONGS_TO', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
                RETURN n.node_id AS id, n.tag AS tag
            UNION
            MATCH (n:Node) WHERE n.val_int > {s._param_int('hi', 700, 999)}
                RETURN n.node_id AS id, n.tag AS tag
        """)

        self._query('set-op-2', 'UNION ALL: Log events and Measure events for a node', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN 'log' AS type, l.log_id AS id, l.occurred_at AS ts, coalesce(l.val, 0.0) AS val
            UNION ALL
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN 'measure' AS type, m.measure_id AS id, datetime({{year: date(m.recorded_at).year, month: date(m.recorded_at).month, day: date(m.recorded_at).day}}) AS ts, m.val AS val
            ORDER BY ts DESC
        """)

        self._query('set-op-3', 'UNION ALL: active node IDs from three different tag groups', lambda s: f"""
            MATCH (n:Node {{tag: '{s._param_tag()}'}}) WHERE n.is_active = true
                RETURN n.node_id AS id
            UNION ALL
            MATCH (n:Node) WHERE n.val_int >= {s._param_int('lo', 1, 300)} AND n.val_int <= {s._param_int('hi', 700, 1000)}
                RETURN n.node_id AS id
            UNION ALL
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{depth: 0}}) WHERE n.status = 0
                RETURN n.node_id AS id
        """)

        self._query('set-op-4', 'UNION ALL chains from Log + Measure + LINKED for a node', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN 'log' AS category, count(l) AS cnt
            UNION ALL
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN 'measure' AS category, count(m) AS cnt
            UNION ALL
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(nb:Node)
                RETURN 'out_link' AS category, count(nb) AS cnt
            UNION ALL
            MATCH (in_n:Node)-[:LINKED]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN 'in_link' AS category, count(in_n) AS cnt
        """)

        #endregion
        #region Window-style computations with COLLECT + UNWIND

        self._query('window-0', 'Running sum via COLLECT + reduce inside a pipeline', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE l.val IS NOT NULL
            WITH l ORDER BY l.occurred_at
            WITH collect({{val: l.val, ts: l.occurred_at}}) AS events
            UNWIND range(0, size(events)-1) AS i
            RETURN events[i].ts AS ts, events[i].val AS val,
                   reduce(s=0.0, j IN range(0,i) | s + events[j].val) AS running_sum
        """)

        self._query('window-1', 'Row number equivalent via collect + index', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WITH n ORDER BY n.val_int DESC
            WITH collect(n) AS nodes
            UNWIND range(0, size(nodes)-1) AS i
            RETURN i+1 AS rank, nodes[i].node_id AS node_id, nodes[i].tag AS tag, nodes[i].val_int AS val_int
            LIMIT 20
        """)

        self._query('window-2', 'Lag equivalent: previous value in sorted sequence', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE m.dim = {s._param_dim()}
            WITH m ORDER BY m.recorded_at
            WITH collect({{val: m.val, ts: m.recorded_at}}) AS recs
            UNWIND range(0, size(recs)-1) AS i
            RETURN recs[i].ts AS ts, recs[i].val AS val,
                   CASE WHEN i > 0 THEN recs[i-1].val ELSE null END AS prev_val,
                   CASE WHEN i > 0 THEN recs[i].val - recs[i-1].val ELSE null END AS delta
        """)

        self._query('window-3', 'StdDev and percentile of val_int per group', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            RETURN count(n) AS cnt,
                   stDev(n.val_int) AS stddev,
                   percentileCont(n.val_int, 0.25) AS p25,
                   percentileCont(n.val_int, 0.75) AS p75
        """)

        self._query('window-4', 'Top-3 per group using COLLECT slice + UNWIND', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 1
            WITH g, n ORDER BY n.val_int DESC
            WITH g, collect(n)[0..3] AS top3
            UNWIND top3 AS tn
            RETURN g.grp_id, g.name, tn.node_id, tn.tag, tn.val_int
        """)

        self._query('window-5', 'Decile bucketing via list size division', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            WITH collect(n.val_int) AS all_vals
            WITH all_vals, size(all_vals) AS n_count
            UNWIND range(0,9) AS bucket
            RETURN bucket,
                   all_vals[toInteger(bucket * n_count / 10)] AS bucket_min,
                   all_vals[toInteger((bucket+1) * n_count / 10) - 1] AS bucket_max
        """)

        #endregion
        #region Graph algorithms in Cypher

        self._query('graph-0', 'Out-degree distribution of Node via LINKED', lambda s: f"""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:LINKED]->(out:Node)
            WITH n.node_id AS node_id, count(out) AS out_degree
            RETURN out_degree, count(*) AS node_cnt
            ORDER BY out_degree
        """)

        self._query('graph-1', 'In-degree vs out-degree with imbalance metric', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            OPTIONAL MATCH (n)-[:LINKED]->(o:Node)
            WITH n, count(DISTINCT o) AS out_deg
            OPTIONAL MATCH (i:Node)-[:LINKED]->(n)
            WITH n, out_deg, count(DISTINCT i) AS in_deg
            RETURN n.node_id, n.tag, out_deg, in_deg, toFloat(out_deg - in_deg) / (out_deg + in_deg + 1) AS imbalance
            ORDER BY abs(out_deg - in_deg) DESC
            LIMIT 20
        """)

        self._query('graph-2', 'Triangle detection: 3-cycle A->B->C->A', lambda s: f"""
            MATCH (a:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(b:Node)-[:LINKED]->(c:Node)-[:LINKED]->(a)
            RETURN DISTINCT b.node_id AS b_id, c.node_id AS c_id
            LIMIT 20
        """)

        self._query('graph-3', 'Common out-neighbors of two nodes', lambda s: f"""
            MATCH (a:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(common:Node)
            MATCH (b:Node {{node_id: {s._param_node_id('node_id_b')}}})-[:LINKED]->(common)
            RETURN DISTINCT common.node_id, common.tag
            LIMIT 50
        """)

        self._query('graph-4', 'Hub detection: nodes with in_degree + out_degree > threshold', lambda s: f"""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:LINKED]->(o:Node)
            WITH n, count(DISTINCT o) AS out_deg
            OPTIONAL MATCH (i:Node)-[:LINKED]->(n)
            WITH n, out_deg, count(DISTINCT i) AS in_deg
            WHERE out_deg + in_deg > {s._param_int('hub_threshold', 5, 30)}
            RETURN n.node_id, n.tag, out_deg, in_deg, out_deg + in_deg AS total_deg
            ORDER BY total_deg DESC
        """)

        self._query('graph-5', 'Neighborhood expansion: 2-hop ego subgraph node count', lambda s: f"""
            MATCH (center:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..2]->(neighbor:Node)
            RETURN count(DISTINCT neighbor) AS ego_size
        """)

        self._query('graph-6', 'Weighted path: max-weight shortest path via UNWIND of paths', lambda s: f"""
            MATCH p = (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..4]->(dst:Node)
            WITH p, reduce(w=0.0, r IN relationships(p) | w + r.weight) AS path_weight
            RETURN [x IN nodes(p) | x.node_id] AS path_ids, path_weight
            ORDER BY path_weight DESC
            LIMIT 5
        """)

        self._query('graph-7', 'Bidirectional 1-hop neighbors (UNION of in + out)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(out:Node)
                RETURN out.node_id AS neighbor_id, out.tag AS tag, 'out' AS dir
            UNION
            MATCH (in_n:Node)-[:LINKED]->(n:Node {{node_id: {s._param_node_id()}}})
                RETURN in_n.node_id AS neighbor_id, in_n.tag AS tag, 'in' AS dir
        """)

        self._query('graph-8', 'Link-kind distribution per node', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(nb:Node)
            RETURN r.kind AS kind, count(*) AS cnt, avg(r.weight) AS avg_weight
            ORDER BY kind
        """)

        self._query('graph-9', 'Full graph degree analysis with z-score', lambda s: f"""
            MATCH (n:Node)
            OPTIONAL MATCH (n)-[:LINKED]->(o:Node)
            WITH n, count(DISTINCT o) AS out_deg
            OPTIONAL MATCH (i:Node)-[:LINKED]->(n)
            WITH n, out_deg, count(DISTINCT i) AS in_deg
            WITH n, out_deg, in_deg, out_deg + in_deg AS total_deg
            WITH collect({{node_id: n.node_id, tag: n.tag, total_deg: total_deg}}) AS all_nodes,
                 avg(toFloat(out_deg + in_deg)) AS mean_deg,
                 stDev(toFloat(out_deg + in_deg)) AS std_deg
            UNWIND all_nodes AS node
            RETURN node.node_id, node.tag, node.total_deg,
                   CASE WHEN std_deg > 0 THEN (node.total_deg - mean_deg) / std_deg ELSE 0.0 END AS z_score
            ORDER BY z_score DESC
            LIMIT 30
        """)

        #endregion
        #region Advanced / complex multi-feature compositions

        self._query('complex-0', 'Grp hierarchy traversal + node stats per level', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})
            OPTIONAL MATCH (child:Grp)-[:CHILD_OF]->(root)
            WITH root, collect(child) AS children
            UNWIND (CASE size(children) > 0 WHEN true THEN children ELSE [root] END) AS g
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WITH g, count(n) AS node_cnt, avg(n.val_int) AS avg_val, collect(n.status) AS statuses
            RETURN g.grp_id, g.name, node_cnt, avg_val,
                   size([s IN statuses WHERE s = 0]) AS active_in_grp
            ORDER BY node_cnt DESC
            LIMIT 20
        """)

        self._query('complex-1', 'Multi-hop node context: node + grp ancestry + recent log', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:BELONGS_TO]->(g:Grp)
            OPTIONAL MATCH (g)-[:CHILD_OF]->(parent:Grp)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, g, parent, l ORDER BY l.occurred_at DESC
            WITH n, g, parent, collect(l)[0] AS latest_log
            RETURN n.node_id, n.tag, g.name AS grp_name,
                   coalesce(parent.name, 'root') AS parent_grp,
                   latest_log.kind AS last_event_kind,
                   latest_log.occurred_at AS last_event_ts
        """)

        self._query('complex-2', 'Top-5 nodes per group by measure avg (COLLECT slice)', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 1
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n)
            WITH g, n, avg(m.val) AS avg_m ORDER BY avg_m IS NULL ASC, avg_m DESC
            WITH g, collect({{node_id: n.node_id, tag: n.tag, avg_m: avg_m}})[0..5] AS top5
            UNWIND top5 AS item
            RETURN g.grp_id, g.name, item.node_id, item.tag, item.avg_m
        """)

        self._query('complex-3', 'Temporal correlation: first/last log and measure per node', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()} AND n.is_active = true
            CALL (n) {{
                OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
                RETURN min(l.occurred_at) AS first_log, max(l.occurred_at) AS last_log, count(l) AS log_cnt
            }}
            CALL (n) {{
                OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN min(m.recorded_at) AS first_m, max(m.recorded_at) AS last_m, count(m) AS m_cnt
            }}
            RETURN n.node_id, n.tag, first_log, last_log, log_cnt, first_m, last_m, m_cnt
        """)

        self._query('complex-4', 'Cascading aggregation: node->dim stats -> group meta-stats', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 0
            WITH n, g, m.dim AS dim, avg(m.val) AS dim_avg, count(m) AS dim_cnt
            WITH g, n, collect({{dim: dim, avg: dim_avg, cnt: dim_cnt}}) AS dim_stats
            UNWIND dim_stats AS ds
            WITH g, n, ds
            WITH g, avg(ds.avg) AS meta_avg, count(DISTINCT n) AS node_cnt
            RETURN g.grp_id, g.name, node_cnt, meta_avg
            ORDER BY meta_avg DESC
        """)

        self._query('complex-5', 'Neighbor link-kind pivot using pattern comprehension per kind', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            WITH n,
                 size([(n)-[r:LINKED]->(nb:Node) WHERE r.kind = 0 | r]) AS k0,
                 size([(n)-[r:LINKED]->(nb:Node) WHERE r.kind = 1 | r]) AS k1,
                 size([(n)-[r:LINKED]->(nb:Node) WHERE r.kind = 2 | r]) AS k2,
                 size([(n)-[r:LINKED]->(nb:Node) WHERE r.kind = 3 | r]) AS k3,
                 size([(n)-[r:LINKED]->(nb:Node) WHERE r.kind = 4 | r]) AS k4
            RETURN n.node_id, n.tag, k0, k1, k2, k3, k4
        """)

        self._query('complex-6', 'Compound: active nodes + doc + above-avg log val', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WHERE n.is_active = true
            MATCH (d:Doc)-[:DOC_OF]->(n)
            WITH n, g, d
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val IS NOT NULL
                WITH avg(l.val) AS node_avg
                RETURN node_avg
            }}
            WITH n, g, d, node_avg
            MATCH (l:Log)-[:EVENT_OF]->(n)
            WHERE l.val > node_avg
            RETURN n.node_id, n.tag, g.name, d.doc_id, node_avg, count(l) AS above_avg_cnt
        """)

        self._query('complex-7', 'COLLECT then UNWIND for cross-product style join', lambda s: f"""
            MATCH (g:Grp {{depth: 0}})
            WITH collect(g.grp_id) AS root_ids
            UNWIND root_ids AS rid
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: rid}})
            WHERE n.is_active = true
            WITH rid, count(n) AS active_cnt, avg(n.val_int) AS avg_val
            RETURN rid, active_cnt, avg_val
            ORDER BY active_cnt DESC
        """)

        self._query('complex-8', 'Linear regression on measure time series via list math', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE m.dim = {s._param_dim()}
            WITH m ORDER BY m.recorded_at
            WITH collect({{val: m.val, idx: 0}}) AS recs, count(m) AS n_count
            UNWIND range(0, size(recs)-1) AS i
            WITH i, recs[i].val AS val, n_count
            WITH count(*) AS cnt, sum(toFloat(i)) AS sx, sum(val) AS sy,
                 sum(toFloat(i) * val) AS sxy, sum(toFloat(i) * toFloat(i)) AS sx2
            RETURN cnt, sx, sy, sxy, sx2,
                   CASE WHEN cnt * sx2 - sx * sx <> 0
                       THEN (cnt * sxy - sx * sy) / (cnt * sx2 - sx * sx)
                       ELSE null
                   END AS slope
        """)

        self._query('complex-9', 'Monster: path + aggregation + conditional + collect', lambda s: f"""
            MATCH (src:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WHERE src.is_active = true
            MATCH (src)-[r:LINKED]->(dst:Node)
            WHERE r.weight > {s._param_weight()}
            WITH src, dst, r,
                 [(l:Log)-[:EVENT_OF]->(src) WHERE l.val IS NOT NULL | l.val] AS src_log_vals
            WITH src, dst, r, src_log_vals,
                 reduce(s=0.0, v IN src_log_vals | s + v) AS src_log_sum
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(dst)
            WHERE m.dim = {s._param_dim()}
            WITH src, dst, r, src_log_sum, avg(m.val) AS dst_measure_avg
            RETURN src.node_id AS src_id, src.tag AS src_tag,
                   dst.node_id AS dst_id, dst.tag AS dst_tag,
                   r.weight AS link_weight,
                   src_log_sum,
                   coalesce(dst_measure_avg, 0.0) AS dst_measure_avg,
                   src_log_sum * r.weight AS composite_score
            ORDER BY composite_score DESC
            LIMIT 20
        """)

        #endregion

    def _register_extended_queries(self):
        # Extended coverage
        # =================
        # grp-scan-0   ... grp-scan-9     Grp-first label scans
        # log-scan-0   ... log-scan-9     Log-first scans
        # measure-scan-0..measure-scan-9  Measure-first scans
        # doc-scan-0   ... doc-scan-4     Doc-first scans
        # deep-join-0  ... deep-join-9    5-7 entity traversals
        # recursive-0  ... recursive-9   CHILD_OF/LINKED recursion
        # selectivity-0... selectivity-19 Threshold spectrum
        # nested-call-0... nested-call-7  2-3 level nested CALL {}
        # fanout-0     ... fanout-7       UNWIND fan-out + re-agg
        # corr-0       ... corr-7         Cross-entity correlation
        # foreach-0    ... foreach-4      FOREACH / UNWIND batch writes
        # extreme-0    ... extreme-9      Hardest full-graph queries

        #region Grp-first scans

        self._query('grp-scan-0', 'Grp full scan: all root groups (depth=0)', lambda s: f"""
            MATCH (g:Grp)
            WHERE g.depth = 0
            RETURN g.grp_id, g.name, g.priority
            ORDER BY g.grp_id
        """)

        self._query('grp-scan-1', 'Grp by depth bucket: depth=1 children', lambda s: f"""
            MATCH (g:Grp)
            WHERE g.depth = 1
            RETURN g.grp_id, g.name, g.priority
            ORDER BY g.name
        """)

        self._query('grp-scan-2', 'Grp by priority range', lambda s: f"""
            MATCH (g:Grp)
            WHERE g.priority >= {s._param_float('lo', 0.1, 0.4, 2)}
              AND g.priority <= {s._param_float('hi', 0.6, 0.99, 2)}
            RETURN g.grp_id, g.name, g.depth, g.priority
            ORDER BY g.priority DESC
        """)

        self._query('grp-scan-3', 'Grp count by depth level', lambda s: f"""
            MATCH (g:Grp)
            RETURN g.depth AS depth, count(g) AS grp_cnt, avg(g.priority) AS avg_priority
            ORDER BY depth
        """)

        self._query('grp-scan-4', 'Grp with node count (Grp + BELONGS_TO join)', lambda s: f"""
            MATCH (g:Grp)
            OPTIONAL MATCH (n:Node)-[:BELONGS_TO]->(g)
            RETURN g.grp_id, g.name, g.depth, count(n) AS node_cnt
            ORDER BY node_cnt DESC
            LIMIT {s._param_limit(20, 3)}
        """)

        self._query('grp-scan-5', 'Grp with active-node ratio', lambda s: f"""
            MATCH (g:Grp)
            OPTIONAL MATCH (n:Node)-[:BELONGS_TO]->(g)
            WITH g, count(n) AS total, count(CASE WHEN n.is_active THEN 1 END) AS active_cnt
            WHERE total > 0
            RETURN g.grp_id, g.name, total, active_cnt,
                   toFloat(active_cnt) / total AS active_ratio
            ORDER BY active_ratio DESC
        """)

        self._query('grp-scan-6', 'Grp with child count (CHILD_OF reverse)', lambda s: f"""
            MATCH (g:Grp)
            OPTIONAL MATCH (child:Grp)-[:CHILD_OF]->(g)
            RETURN g.grp_id, g.name, g.depth, count(child) AS child_cnt
            ORDER BY child_cnt DESC
        """)

        self._query('grp-scan-7', 'Grp name starts-with prefix filter', lambda s: f"""
            MATCH (g:Grp)
            WHERE g.name STARTS WITH '{s._param_tag()[0]}'
            RETURN g.grp_id, g.name, g.depth
            ORDER BY g.name
            LIMIT 50
        """)

        self._query('grp-scan-8', 'Grp priority top-K with node stats', lambda s: f"""
            MATCH (g:Grp)
            WITH g ORDER BY g.priority DESC
            LIMIT {s._param_limit(10, 2)}
            OPTIONAL MATCH (n:Node)-[:BELONGS_TO]->(g)
            RETURN g.grp_id, g.name, g.priority, count(n) AS node_cnt, avg(n.val_int) AS avg_val
        """)

        self._query('grp-scan-9', 'Grp HAVING at least N active nodes', lambda s: f"""
            MATCH (g:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WHERE n.is_active = true
            WITH g, count(n) AS active_cnt
            WHERE active_cnt >= {s._param_int('min_active', 5, 50)}
            RETURN g.grp_id, g.name, g.depth, active_cnt
            ORDER BY active_cnt DESC
        """)

        #endregion
        #region Log-first scans

        self._query('log-scan-0', 'Log full scan: count and val stats', lambda s: f"""
            MATCH (l:Log)
            RETURN count(l) AS total, avg(l.val) AS avg_val,
                   min(l.val) AS min_val, max(l.val) AS max_val
        """)

        self._query('log-scan-1', 'Log by kind: count per kind', lambda s: f"""
            MATCH (l:Log)
            RETURN l.kind AS kind, count(l) AS cnt, avg(l.val) AS avg_val
            ORDER BY kind
        """)

        self._query('log-scan-2', 'Log val > threshold (high-selectivity scan)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val > {s._param_float('threshold', 80.0, 99.0)}
            RETURN l.log_id, l.kind, l.val, l.occurred_at, l.node_id
            ORDER BY l.val DESC
            LIMIT 100
        """)

        self._query('log-scan-3', 'Log occurred_at range filter', lambda s: f"""
            MATCH (l:Log)
            WHERE l.occurred_at >= datetime('{s._param_date_minus_days(7, 30)}')
              AND l.occurred_at <  datetime('{s._param_date_minus_days(0, 6)}')
            RETURN l.log_id, l.node_id, l.kind, l.val, l.occurred_at
            ORDER BY l.occurred_at DESC
            LIMIT 200
        """)

        self._query('log-scan-4', 'Log by kind + val range (2-column filter)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.kind = {s._param_log_kind()}
              AND l.val >= {s._param_float('lo', 10.0, 40.0)} AND l.val <= {s._param_float('hi', 60.0, 90.0)}
            RETURN l.log_id, l.node_id, l.val, l.occurred_at
            LIMIT 100
        """)

        self._query('log-scan-5', 'Log null-val rows', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val IS NULL
              AND l.kind = {s._param_log_kind()}
            RETURN l.log_id, l.node_id, l.occurred_at
            LIMIT 100
        """)

        self._query('log-scan-6', 'Log per-node aggregation (node_id GROUP BY)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.node_id = {s._param_node_id()}
            RETURN l.kind AS kind, count(l) AS cnt,
                   avg(l.val) AS avg_val, max(l.val) AS max_val,
                   min(l.occurred_at) AS first_ts, max(l.occurred_at) AS last_ts
            ORDER BY kind
        """)

        self._query('log-scan-7', 'Log daily histogram via date truncation', lambda s: f"""
            MATCH (l:Log)
            WHERE l.occurred_at >= datetime('{s._param_date_minus_days(30, 90)}')
              AND l.val IS NOT NULL
            WITH date(l.occurred_at) AS day, count(l) AS cnt, avg(l.val) AS avg_val
            RETURN day, cnt, avg_val
            ORDER BY day
        """)

        self._query('log-scan-8', 'Log top-10 by val per kind (COLLECT slice)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val IS NOT NULL
            WITH l.kind AS kind, l ORDER BY l.val DESC
            WITH kind, collect(l)[0..10] AS top_logs
            UNWIND top_logs AS tl
            RETURN kind, tl.log_id, tl.node_id, tl.val, tl.occurred_at
        """)

        self._query('log-scan-9', 'Log outliers: val > mean + 2*stdev', lambda s: f"""
            MATCH (l:Log)
            WHERE l.val IS NOT NULL AND l.kind = {s._param_log_kind()}
            WITH avg(l.val) AS mean_val, stDev(l.val) AS std_val
            MATCH (l2:Log)
            WHERE l2.kind = {s._param_log_kind()}
              AND l2.val IS NOT NULL
              AND l2.val > mean_val + 2 * std_val
            RETURN l2.log_id, l2.node_id, l2.val, mean_val, std_val
            ORDER BY l2.val DESC
            LIMIT 50
        """)

        #endregion
        #region Measure-first scans

        self._query('measure-scan-0', 'Measure full scan: count and val stats per dim', lambda s: f"""
            MATCH (m:Measure)
            RETURN m.dim AS dim, count(m) AS cnt,
                   avg(m.val) AS avg_val, min(m.val) AS min_val, max(m.val) AS max_val
            ORDER BY dim
        """)

        self._query('measure-scan-1', 'Measure by dim + val range', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.dim = {s._param_dim()}
              AND m.val >= {s._param_float('lo', 5.0, 30.0)} AND m.val <= {s._param_float('hi', 70.0, 95.0)}
            RETURN m.measure_id, m.node_id, m.val, m.recorded_at
            ORDER BY m.val DESC
            LIMIT 100
        """)

        self._query('measure-scan-2', 'Measure recorded_at range filter', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.recorded_at >= date('{s._param_date_minus_days(30, 180)}')
              AND m.dim = {s._param_dim()}
            RETURN m.measure_id, m.node_id, m.dim, m.val, m.recorded_at
            ORDER BY m.recorded_at DESC
            LIMIT 200
        """)

        self._query('measure-scan-3', 'Measure per-node aggregation', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.node_id = {s._param_node_id()}
            RETURN m.dim AS dim, count(m) AS cnt,
                   avg(m.val) AS avg_val, percentileCont(m.val, 0.5) AS median_val
            ORDER BY dim
        """)

        self._query('measure-scan-4', 'Measure top-5 vals per dim', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.val IS NOT NULL
            WITH m.dim AS dim, m ORDER BY m.val DESC
            WITH dim, collect(m)[0..5] AS top_m
            UNWIND top_m AS tm
            RETURN dim, tm.measure_id, tm.node_id, tm.val, tm.recorded_at
        """)

        self._query('measure-scan-5', 'Measure null-val rows per dim', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.val IS NULL
            RETURN m.dim AS dim, count(m) AS null_cnt
            ORDER BY dim
        """)

        self._query('measure-scan-6', 'Measure daily avg per dim time series', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.dim = {s._param_dim()}
              AND m.recorded_at >= date('{s._param_date_minus_days(60, 180)}')
            WITH m.recorded_at AS day, avg(m.val) AS avg_val, count(m) AS cnt
            RETURN day, avg_val, cnt
            ORDER BY day
        """)

        self._query('measure-scan-7', 'Measure outliers: val < p5 or val > p95 (CTE-style)', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.dim = {s._param_dim()} AND m.val IS NOT NULL
            WITH collect(m.val) AS all_vals
            WITH all_vals,
                 all_vals[toInteger(size(all_vals) * 0.05)] AS p5,
                 all_vals[toInteger(size(all_vals) * 0.95)] AS p95
            MATCH (m2:Measure)
            WHERE m2.dim = {s._param_dim()} AND m2.val IS NOT NULL
              AND (m2.val < p5 OR m2.val > p95)
            RETURN m2.measure_id, m2.node_id, m2.val, p5, p95
            ORDER BY m2.val
            LIMIT 100
        """)

        self._query('measure-scan-8', 'Measure stdev per node (only nodes with >= 3 measures)', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)
            WHERE m.dim = {s._param_dim()} AND m.val IS NOT NULL
            WITH n, count(m) AS cnt, stDev(m.val) AS stddev_val
            WHERE cnt >= 3
            RETURN n.node_id, n.tag, cnt, stddev_val
            ORDER BY stddev_val DESC
            LIMIT 50
        """)

        self._query('measure-scan-9', 'Measure moving average via COLLECT + UNWIND window', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE m.dim = {s._param_dim()} AND m.val IS NOT NULL
            WITH m ORDER BY m.recorded_at
            WITH collect({{val: m.val, day: m.recorded_at}}) AS recs
            UNWIND range(0, size(recs)-1) AS i
            WITH i, recs, recs[i] AS cur,
                 reduce(s=0.0, j IN range(CASE WHEN i<2 THEN 0 ELSE i-2 END, i) | s + recs[j].val)
                     / toFloat(CASE WHEN i<2 THEN i+1 ELSE 3 END) AS ma3
            RETURN cur.day AS day, cur.val AS val, ma3
            ORDER BY day
        """)

        #endregion
        #region Doc-first scans

        self._query('doc-scan-0', 'Doc full scan: count and date range', lambda s: f"""
            MATCH (d:Doc)
            RETURN count(d) AS total,
                   min(d.created_at) AS oldest, max(d.created_at) AS newest
        """)

        self._query('doc-scan-1', 'Doc by created_at range', lambda s: f"""
            MATCH (d:Doc)
            WHERE d.created_at >= datetime('{s._param_date_minus_days(30, 180)}')
            RETURN d.doc_id, d.node_id, d.created_at
            ORDER BY d.created_at DESC
            LIMIT 100
        """)

        self._query('doc-scan-2', 'Doc + Node join: body length + node tag', lambda s: f"""
            MATCH (d:Doc)-[:DOC_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            RETURN d.doc_id, n.node_id, n.tag, n.status,
                   size(d.body) AS body_len, d.created_at
            ORDER BY body_len DESC
            LIMIT 50
        """)

        self._query('doc-scan-3', 'Doc coverage: nodes with and without Doc per group', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE g.depth = 1
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            WITH g, count(n) AS total, count(d) AS with_doc
            RETURN g.grp_id, g.name, total, with_doc,
                   total - with_doc AS without_doc,
                   toFloat(with_doc) / total AS doc_ratio
            ORDER BY doc_ratio
        """)

        self._query('doc-scan-4', 'Doc created per day histogram', lambda s: f"""
            MATCH (d:Doc)
            WHERE d.created_at >= datetime('{s._param_date_minus_days(60, 180)}')
            WITH date(d.created_at) AS day, count(d) AS cnt
            RETURN day, cnt
            ORDER BY day
        """)

        #endregion
        #region Deep joins (5-7 entities)

        self._query('deep-join-0', '5-entity: Node+Grp+Log+Measure+Doc star', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            MATCH (l:Log)-[:EVENT_OF]->(n)
            MATCH (m:Measure)-[:METRIC_OF]->(n)
            MATCH (d:Doc)-[:DOC_OF]->(n)
            WHERE n.is_active = true
              AND m.dim = {s._param_dim()}
              AND l.val IS NOT NULL
            RETURN n.node_id, g.name, l.val, m.val AS measure_val, d.doc_id
            LIMIT 50
        """)

        self._query('deep-join-1', '6-entity: Node+Grp+parentGrp+Log+Measure+Doc', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)-[:CHILD_OF]->(pg:Grp)
            MATCH (m:Measure)-[:METRIC_OF]->(n)
            MATCH (d:Doc)-[:DOC_OF]->(n)
            WHERE n.is_active = true AND m.dim = {s._param_dim()} AND l.val IS NOT NULL
            RETURN n.node_id, g.name, pg.name AS parent_grp,
                   l.val, m.val AS measure_val, d.doc_id
            LIMIT 30
        """)

        self._query('deep-join-2', '6-entity: LINKED pair + Grp + Log + Measure + Doc', lambda s: f"""
            MATCH (src:Node)-[r:LINKED]->(dst:Node)
            MATCH (src)-[:BELONGS_TO]->(sg:Grp)
            MATCH (l:Log)-[:EVENT_OF]->(src)
            MATCH (m:Measure)-[:METRIC_OF]->(dst)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(src)
            WHERE r.weight > {s._param_weight()}
              AND m.dim = {s._param_dim()}
            RETURN src.node_id, sg.name, dst.node_id, r.weight,
                   l.val AS log_val, m.val AS measure_val, d.doc_id
            LIMIT 30
        """)

        self._query('deep-join-3', '7-entity: full star with parent hierarchy', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)-[:CHILD_OF]->(pg:Grp)
            MATCH (m:Measure)-[:METRIC_OF]->(n)
            MATCH (d:Doc)-[:DOC_OF]->(n)
            MATCH (n)-[r:LINKED]->(nb:Node)
            WHERE n.status = {s._param_status()} AND m.dim = {s._param_dim()}
            RETURN n.node_id, g.name, pg.name AS parent_grp,
                   l.val, m.val AS mval, d.doc_id, nb.node_id AS neighbor, r.weight
            LIMIT 20
        """)

        self._query('deep-join-4', '5-entity agg: Grp+parentGrp+Node+Log+Measure with group stats', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)-[:CHILD_OF]->(pg:Grp)
            MATCH (l:Log)-[:EVENT_OF]->(n)
            WHERE pg.depth = 0 AND m.dim = {s._param_dim()} AND l.val IS NOT NULL
            RETURN pg.grp_id, pg.name, g.grp_id, g.name,
                   count(DISTINCT n) AS node_cnt,
                   avg(m.val) AS avg_measure,
                   avg(l.val) AS avg_log
            ORDER BY avg_measure DESC
        """)

        self._query('deep-join-5', '5-entity: LINKED chain src->mid->dst + grp + doc', lambda s: f"""
            MATCH (src:Node)-[:LINKED]->(mid:Node)-[:LINKED]->(dst:Node)
            MATCH (src)-[:BELONGS_TO]->(g:Grp)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(dst)
            WHERE src.is_active = true AND g.depth = 1
            RETURN src.node_id, mid.node_id, dst.node_id,
                   g.name, d.doc_id
            LIMIT 30
        """)

        self._query('deep-join-6', '6-entity: 2-hop LINKED chain + both endpoint Grps + Log', lambda s: f"""
            MATCH (src:Node)-[:LINKED]->(mid:Node)-[:LINKED]->(dst:Node)
            MATCH (src)-[:BELONGS_TO]->(sg:Grp)
            MATCH (dst)-[:BELONGS_TO]->(dg:Grp)
            MATCH (l:Log)-[:EVENT_OF]->(mid)
            WHERE l.val IS NOT NULL
            RETURN src.node_id, mid.node_id, dst.node_id,
                   sg.name AS src_grp, dg.name AS dst_grp,
                   l.val AS mid_log_val
            LIMIT 30
        """)

        self._query('deep-join-7', '5-entity: Grp+children+Nodes+Measures+agg per child', lambda s: f"""
            MATCH (pg:Grp {{grp_id: {s._param_grp_id()}}})<-[:CHILD_OF]-(cg:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(cg)
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n)
            WHERE m.dim = {s._param_dim()} OR m IS NULL
            RETURN pg.name, cg.grp_id, cg.name,
                   count(DISTINCT n) AS node_cnt,
                   avg(m.val) AS avg_measure
            ORDER BY avg_measure IS NULL ASC, avg_measure DESC
        """)

        self._query('deep-join-8', '6-entity: LINKED neighborhood + Log + Measure + Doc per neighbor', lambda s: f"""
            MATCH (center:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(nb:Node)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(nb)
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(nb)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(nb)
            WITH nb, count(l) AS log_cnt, avg(m.val) AS avg_m, d.doc_id AS doc_id
            RETURN nb.node_id, nb.tag, log_cnt, avg_m, doc_id
            ORDER BY log_cnt DESC
            LIMIT 20
        """)

        self._query('deep-join-9', '7-entity: full-chain with agg rollup at every level', lambda s: f"""
            MATCH (pg:Grp {{depth: 0}})<-[:CHILD_OF]-(cg:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(cg)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n)
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            OPTIONAL MATCH (n)-[:LINKED]->(nb:Node)
            WITH pg, cg, n,
                 count(l) AS log_cnt, avg(m.val) AS avg_m,
                 count(d) AS doc_cnt, count(nb) AS out_deg
            WITH pg, cg,
                 count(n) AS node_cnt,
                 sum(log_cnt) AS total_logs,
                 avg(avg_m) AS meta_avg_m,
                 sum(doc_cnt) AS total_docs,
                 avg(out_deg) AS avg_out_deg
            RETURN pg.grp_id, pg.name, cg.grp_id, cg.name,
                   node_cnt, total_logs, meta_avg_m, total_docs, avg_out_deg
            ORDER BY total_logs DESC
        """)

        #endregion
        #region Recursive / variable-depth traversals

        self._query('recursive-0', 'CHILD_OF*1..8 all ancestors with depth', lambda s: f"""
            MATCH path = (leaf:Grp)-[:CHILD_OF*1..8]->(root:Grp)
            WHERE NOT EXISTS {{ MATCH (root)-[:CHILD_OF]->(:Grp) }}
            WITH root, length(path) AS depth, leaf
            RETURN root.grp_id, root.name, depth, count(leaf) AS leaf_cnt
            ORDER BY depth, root.grp_id
        """)

        self._query('recursive-1', 'Full subtree of a Grp: all descendants via CHILD_OF reverse', lambda s: f"""
            MATCH (root:Grp {{grp_id: {s._param_grp_id()}}})<-[:CHILD_OF*0..8]-(desc:Grp)
            WITH desc, length(shortestPath((desc)-[:CHILD_OF*0..8]->(root))) AS depth
            RETURN desc.grp_id, desc.name, depth
            ORDER BY depth, desc.grp_id
        """)

        self._query('recursive-2', 'Node count at each hierarchy depth level', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})<-[:CHILD_OF*0..5]-(g:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WITH root, g.depth AS level, count(n) AS node_cnt
            RETURN root.grp_id, root.name, level, sum(node_cnt) AS total_nodes
            ORDER BY level
        """)

        self._query('recursive-3', 'All ancestors of a node (via Grp CHILD_OF chain)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[:BELONGS_TO]->(g:Grp)
            MATCH anc_path = (g)-[:CHILD_OF*0..8]->(anc:Grp)
            RETURN DISTINCT anc.grp_id, anc.name, anc.depth, length(anc_path) AS dist
            ORDER BY dist
        """)

        self._query('recursive-4', 'LINKED*1..6 reachability from a node with depth count', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..6]->(dst:Node)
            WITH dst, min(length(shortestPath((src)-[:LINKED*1..6]->(dst)))) AS min_hops
            RETURN min_hops, count(dst) AS reach_cnt
            ORDER BY min_hops
        """)

        self._query('recursive-5', 'Deepest path in LINKED graph from a source', lambda s: f"""
            MATCH p = (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..8]->(dst:Node)
            WITH p, length(p) AS depth
            ORDER BY depth DESC
            LIMIT 1
            RETURN depth, [x IN nodes(p) | x.node_id] AS path_ids,
                   [r IN relationships(p) | r.weight] AS weights
        """)

        self._query('recursive-6', 'Subtree aggregate: total log count under each root', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})<-[:CHILD_OF*0..5]-(g:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            MATCH (l:Log)-[:EVENT_OF]->(n)
            WHERE l.val IS NOT NULL
            RETURN root.grp_id, root.name, count(DISTINCT g) AS sub_grp_cnt,
                   count(DISTINCT n) AS total_nodes, count(l) AS total_logs,
                   avg(l.val) AS avg_log_val
        """)

        self._query('recursive-7', 'Sibling groups (same parent) with node stats', lambda s: f"""
            MATCH (g:Grp {{grp_id: {s._param_grp_id()}}})-[:CHILD_OF]->(parent:Grp)
            MATCH (sib:Grp)-[:CHILD_OF]->(parent)
            WHERE sib <> g
            OPTIONAL MATCH (n:Node)-[:BELONGS_TO]->(sib)
            RETURN parent.grp_id, sib.grp_id, sib.name,
                   count(n) AS node_cnt, avg(n.val_int) AS avg_val
            ORDER BY node_cnt DESC
        """)

        self._query('recursive-8', 'LINKED*2 friends-of-friends excluding direct neighbors', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED]->(direct:Node)
            WITH src, collect(DISTINCT direct.node_id) AS direct_ids
            MATCH (src)-[:LINKED*2]->(fof:Node)
            WHERE NOT fof.node_id IN direct_ids AND fof <> src
            RETURN DISTINCT fof.node_id, fof.tag
            LIMIT 50
        """)

        self._query('recursive-9', 'Hierarchy depth distribution: how deep is each Grp', lambda s: f"""
            MATCH (leaf:Grp)
            WHERE NOT EXISTS {{ MATCH (leaf)<-[:CHILD_OF]-(:Grp) }}
            MATCH p = (leaf)-[:CHILD_OF*0..10]->(root:Grp)
            WHERE NOT EXISTS {{ MATCH (root)-[:CHILD_OF]->(:Grp) }}
            WITH leaf, max(length(p)) AS max_depth
            RETURN max_depth, count(leaf) AS leaf_cnt
            ORDER BY max_depth
        """)

        #endregion
        #region Selectivity spectrum (systematically varied thresholds via loops)

        # Log val thresholds: ultra-low to ultra-high (selectivity-0..4)
        for i, thr in enumerate([1.0, 10.0, 50.0, 80.0, 95.0]):
            self._query(
                f'selectivity-{i}',
                f'Log val > {thr} (selectivity tier {i})',
                lambda s, t=thr: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE l.val > {t}
              AND n.is_active = true
            RETURN l.log_id, n.node_id, n.tag, l.val, l.occurred_at
            ORDER BY l.val DESC
            LIMIT 100
        """)

        # Measure val thresholds (selectivity-5..9)
        for i, thr in enumerate([1.0, 20.0, 50.0, 75.0, 95.0]):
            self._query(
                f'selectivity-{i + 5}',
                f'Measure val > {thr} (selectivity tier {i})',
                lambda s, t=thr: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)
            WHERE m.val > {t}
              AND m.dim = {s._param_dim()}
            RETURN m.measure_id, n.node_id, n.tag, m.val, m.recorded_at
            ORDER BY m.val DESC
            LIMIT 100
        """)

        # Node val_int thresholds (selectivity-10..14)
        for i, thr in enumerate([50, 200, 500, 750, 950]):
            self._query(
                f'selectivity-{i + 10}',
                f'Node val_int > {thr} (selectivity tier {i})',
                lambda s, t=thr: f"""
            MATCH (n:Node)
            WHERE n.val_int > {t}
              AND n.is_active = true
            RETURN n.node_id, n.tag, n.val_int, n.grp_id
            ORDER BY n.val_int DESC
            LIMIT 100
        """)

        # LINKED weight thresholds (selectivity-15..19)
        for i, thr in enumerate([0.05, 0.2, 0.5, 0.7, 0.9]):
            self._query(
                f'selectivity-{i + 15}',
                f'LINKED weight > {thr} (selectivity tier {i})',
                lambda s, t=thr: f"""
            MATCH (src:Node)-[r:LINKED]->(dst:Node)
            WHERE r.weight > {t}
            RETURN src.node_id, dst.node_id, r.kind, r.weight
            ORDER BY r.weight DESC
            LIMIT 100
        """)

        #endregion
        #region Nested CALL {} (2-3 levels deep)

        self._query('nested-call-0', 'Two-level nested CALL: per-node log agg then group agg', lambda s: f"""
            MATCH (g:Grp {{grp_id: {s._param_grp_id()}}})
            CALL (g) {{
                MATCH (n:Node)-[:BELONGS_TO]->(g)
                CALL (n) {{
                    MATCH (l:Log)-[:EVENT_OF]->(n)
                    WHERE l.val IS NOT NULL
                    RETURN count(l) AS log_cnt, avg(l.val) AS avg_val
                }}
                RETURN n, log_cnt, avg_val
            }}
            RETURN g.grp_id, n.node_id, n.tag, log_cnt, avg_val
            ORDER BY avg_val DESC
            LIMIT 20
        """)

        self._query('nested-call-1', 'Two-level nested CALL: measure stats inside node loop inside grp loop', lambda s: f"""
            MATCH (g:Grp {{depth: 1}})
            CALL (g) {{
                MATCH (n:Node)-[:BELONGS_TO]->(g)
                WHERE n.is_active = true
                CALL (n) {{
                    MATCH (m:Measure)-[:METRIC_OF]->(n)
                    WHERE m.dim = {s._param_dim()}
                    RETURN avg(m.val) AS avg_m, count(m) AS m_cnt
                }}
                RETURN n, avg_m, m_cnt
            }}
            WITH g, n, avg_m, m_cnt
            WHERE m_cnt > 0
            RETURN g.grp_id, g.name, n.node_id, n.tag, avg_m, m_cnt
            ORDER BY avg_m DESC
            LIMIT 30
        """)

        self._query('nested-call-2', 'Two-level CALL: neighbor count inside node loop inside grp', lambda s: f"""
            MATCH (g:Grp {{grp_id: {s._param_grp_id()}}})
            CALL (g) {{
                MATCH (n:Node)-[:BELONGS_TO]->(g)
                CALL (n) {{
                    OPTIONAL MATCH (n)-[:LINKED]->(nb:Node)
                    RETURN count(nb) AS out_deg
                }}
                RETURN n, out_deg
            }}
            RETURN g.grp_id, n.node_id, n.tag, out_deg
            ORDER BY out_deg DESC
            LIMIT 25
        """)

        self._query('nested-call-3', 'Three-level CALL: root->child->node->log agg chain', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})
            CALL (root) {{
                MATCH (cg:Grp)-[:CHILD_OF]->(root)
                CALL (cg) {{
                    MATCH (n:Node)-[:BELONGS_TO]->(cg)
                    CALL (n) {{
                        MATCH (l:Log)-[:EVENT_OF]->(n)
                        WHERE l.val IS NOT NULL
                        RETURN sum(l.val) AS log_sum, count(l) AS log_cnt
                    }}
                    RETURN n, log_sum, log_cnt
                }}
                RETURN cg, sum(log_sum) AS cg_log_sum, sum(log_cnt) AS cg_log_cnt
            }}
            RETURN root.grp_id, root.name, cg.grp_id, cg.name, cg_log_sum, cg_log_cnt
            ORDER BY cg_log_sum DESC
            LIMIT 20
        """)

        self._query('nested-call-4', 'CALL with UNION inside (per-node combined event count)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()} AND n.is_active = true
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                RETURN count(l) AS cnt, 'log' AS kind
              UNION ALL
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN count(m) AS cnt, 'measure' AS kind
            }}
            RETURN n.node_id, n.tag, kind, cnt
            ORDER BY n.node_id, kind
        """)

        self._query('nested-call-5', 'CALL inside EXISTS-like filter for complex correlated exists', lambda s: f"""
            MATCH (g:Grp)-[:CHILD_OF]->(pg:Grp {{depth: 0}})
            CALL (g) {{
                MATCH (n:Node)-[:BELONGS_TO]->(g)
                WHERE n.is_active = true
                CALL (n) {{
                    MATCH (l:Log)-[:EVENT_OF]->(n)
                    WHERE l.val > {s._param_float('hi', 70.0, 95.0)}
                    RETURN count(l) AS hi_log_cnt
                }}
                RETURN sum(hi_log_cnt) AS grp_hi_log_sum
            }}
            WITH * WHERE grp_hi_log_sum > 0
            RETURN pg.grp_id, g.grp_id, g.name, grp_hi_log_sum
            ORDER BY grp_hi_log_sum DESC
        """)

        self._query('nested-call-6', 'Two-level CALL with max+argmax: best measure node per group', lambda s: f"""
            MATCH (g:Grp {{depth: 1}})
            CALL (g) {{
                MATCH (n:Node)-[:BELONGS_TO]->(g)
                CALL (n) {{
                    MATCH (m:Measure)-[:METRIC_OF]->(n)
                    WHERE m.dim = {s._param_dim()}
                    RETURN max(m.val) AS max_m
                }}
                RETURN n, max_m ORDER BY max_m DESC LIMIT 1
            }}
            RETURN g.grp_id, g.name, n.node_id, n.tag, max_m
            ORDER BY max_m DESC
        """)

        self._query('nested-call-7', 'Two-level CALL: per-node doc presence + log stats', lambda s: f"""
            MATCH (n:Node)
            WHERE n.status = {s._param_status()} AND n.is_active = true
            CALL (n) {{
                OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
                RETURN count(d) AS doc_cnt
            }}
            CALL (n) {{
                MATCH (l:Log)-[:EVENT_OF]->(n) WHERE l.val IS NOT NULL
                RETURN avg(l.val) AS avg_log, count(l) AS log_cnt
            }}
            RETURN n.node_id, n.tag, doc_cnt, avg_log, log_cnt
            ORDER BY log_cnt DESC
            LIMIT 30
        """)

        #endregion
        #region Fan-out queries (UNWIND-based explosion + re-aggregation)

        self._query('fanout-0', 'UNWIND grp_ids -> per-grp node stats', lambda s: f"""
            WITH [{s._param_grp_ids(5, 15)}] AS grp_ids
            UNWIND grp_ids AS gid
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: gid}})
            RETURN gid, count(n) AS node_cnt,
                   avg(n.val_int) AS avg_val, sum(n.val_int) AS sum_val
            ORDER BY gid
        """)

        self._query('fanout-1', 'UNWIND dims -> per-dim measure stats', lambda s: f"""
            WITH [0, 1, 2, 3, 4] AS dims
            UNWIND dims AS d
            MATCH (m:Measure)
            WHERE m.dim = d
            RETURN d AS dim, count(m) AS cnt,
                   avg(m.val) AS avg_val, stDev(m.val) AS std_val
            ORDER BY dim
        """)

        self._query('fanout-2', 'UNWIND node_ids -> batch node fetch + log count', lambda s: f"""
            WITH [{s._param_node_ids(10, 30)}] AS ids
            UNWIND ids AS nid
            MATCH (n:Node {{node_id: nid}})
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            RETURN n.node_id, n.tag, n.status, count(l) AS log_cnt
            ORDER BY log_cnt DESC
        """)

        self._query('fanout-3', 'UNWIND statuses -> pivot count per status per group', lambda s: f"""
            WITH [0, 1, 2, 3, 4] AS statuses
            MATCH (g:Grp {{grp_id: {s._param_grp_id()}}})
            UNWIND statuses AS st
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WHERE n.status = st
            RETURN st AS status, count(n) AS cnt
            ORDER BY status
        """)

        self._query('fanout-4', 'UNWIND log kinds -> event frequency per kind per node', lambda s: f"""
            WITH [0, 1, 2, 3, 4, 5, 6, 7] AS kinds
            UNWIND kinds AS k
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WHERE l.kind = k
            RETURN k AS kind, count(l) AS cnt, avg(l.val) AS avg_val
            ORDER BY kind
        """)

        self._query('fanout-5', 'Explode neighbor list then re-aggregate by destination group', lambda s: f"""
            MATCH (src:Node)-[:LINKED]->(dst:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE src.grp_id = {s._param_grp_id()}
            WITH g, collect(DISTINCT dst.node_id) AS dst_ids
            RETURN g.grp_id, g.name, size(dst_ids) AS link_target_cnt, dst_ids[0..5] AS sample_ids
            ORDER BY link_target_cnt DESC
        """)

        self._query('fanout-6', 'UNWIND date offsets -> daily log counts (table-valued generator)', lambda s: f"""
            WITH range(0, 6) AS offsets
            UNWIND offsets AS offset_days
            WITH date() - duration({{days: offset_days}}) AS target_day
            MATCH (l:Log)
            WHERE date(l.occurred_at) = target_day AND l.val IS NOT NULL
            RETURN target_day AS day, count(l) AS log_cnt, avg(l.val) AS avg_val
            ORDER BY day DESC
        """)

        self._query('fanout-7', 'UNWIND edges -> weighted adjacency list for a node ego', lambda s: f"""
            MATCH (center:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(nb:Node)
            WITH collect({{dst: nb.node_id, tag: nb.tag, w: r.weight, kind: r.kind}}) AS edges
            UNWIND edges AS e
            RETURN e.dst AS neighbor_id, e.tag AS neighbor_tag,
                   e.w AS weight, e.kind AS link_kind
            ORDER BY e.w DESC
        """)

        #endregion
        #region Cross-entity correlation queries

        self._query('corr-0', 'Log val vs Measure val per node (join-based correlation input)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)<-[:METRIC_OF]-(m:Measure)
            WHERE n.is_active = true
              AND l.val IS NOT NULL AND m.dim = {s._param_dim()}
            WITH n.node_id AS node_id, avg(l.val) AS avg_log, avg(m.val) AS avg_measure
            RETURN node_id, avg_log, avg_measure,
                   avg_log - avg_measure AS diff
            ORDER BY abs(avg_log - avg_measure) DESC
            LIMIT 50
        """)

        self._query('corr-1', 'Log count vs out-degree: activity vs connectivity', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, count(l) AS log_cnt
            OPTIONAL MATCH (n)-[:LINKED]->(nb:Node)
            WITH n, log_cnt, count(nb) AS out_deg
            RETURN n.node_id, n.tag, log_cnt, out_deg,
                   toFloat(log_cnt) / (out_deg + 1) AS log_per_link
            ORDER BY log_per_link DESC
            LIMIT 50
        """)

        self._query('corr-2', 'Measure avg vs node val_int (property vs derived stat)', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)
            WHERE m.dim = {s._param_dim()} AND m.val IS NOT NULL
            WITH n, avg(m.val) AS avg_m
            RETURN n.node_id, n.tag, n.val_int, avg_m,
                   n.val_int - toInteger(avg_m) AS int_vs_measure_delta
            ORDER BY abs(n.val_int - toInteger(avg_m)) DESC
            LIMIT 50
        """)

        self._query('corr-3', 'Doc presence vs log activity: documented nodes are more active?', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, count(d) AS doc_cnt, count(l) AS log_cnt
            RETURN
                count(CASE WHEN doc_cnt > 0 THEN 1 END) AS nodes_with_doc,
                avg(CASE WHEN doc_cnt > 0 THEN log_cnt END) AS avg_log_with_doc,
                avg(CASE WHEN doc_cnt = 0 THEN log_cnt END) AS avg_log_without_doc
        """)

        self._query('corr-4', 'Group-level: avg log val vs avg measure val correlation', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n) WHERE l.val IS NOT NULL
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n) WHERE m.dim = {s._param_dim()}
            WITH g, avg(l.val) AS avg_log_val, avg(m.val) AS avg_measure_val
            WHERE avg_log_val IS NOT NULL AND avg_measure_val IS NOT NULL
            RETURN g.grp_id, g.name, avg_log_val, avg_measure_val,
                   avg_log_val / (avg_measure_val + 0.001) AS ratio
            ORDER BY ratio DESC
            LIMIT 30
        """)

        self._query('corr-5', 'Node age (days since creation) vs log count (temporal correlation)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.is_active = true
            WITH n, duration.between(date(n.created_at), date()).days AS age_days
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n)
            WITH n, age_days, count(l) AS log_cnt
            RETURN n.node_id, n.tag, age_days, log_cnt,
                   toFloat(log_cnt) / (age_days + 1) AS logs_per_day
            ORDER BY logs_per_day DESC
            LIMIT 50
        """)

        self._query('corr-6', 'Link weight vs destination measure val (edge weight vs endpoint quality)', lambda s: f"""
            MATCH (src:Node)-[r:LINKED]->(dst:Node)
            WHERE src.grp_id = {s._param_grp_id()}
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(dst)
            WHERE m.dim = {s._param_dim()}
            WITH r.weight AS weight, avg(m.val) AS dst_avg_measure
            WHERE dst_avg_measure IS NOT NULL
            RETURN weight, dst_avg_measure,
                   weight * dst_avg_measure AS weighted_quality
            ORDER BY weighted_quality DESC
            LIMIT 50
        """)

        self._query('corr-7', 'Cross-group: avg val_int per group vs parent priority', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp)-[:CHILD_OF]->(pg:Grp)
            WITH pg, g, avg(n.val_int) AS avg_node_val, count(n) AS node_cnt
            RETURN pg.grp_id, pg.priority, g.grp_id, g.name,
                   avg_node_val, node_cnt,
                   pg.priority * avg_node_val AS priority_weighted_val
            ORDER BY priority_weighted_val DESC
            LIMIT 30
        """)

        #endregion
        #region Extreme / hardest queries

        self._query('extreme-0', 'Full graph scan: all-pairs reachability summary via LINKED*1..3', lambda s: f"""
            MATCH (src:Node)
            WHERE src.is_active = true
            MATCH (src)-[:LINKED*1..3]->(dst:Node)
            WITH src.node_id AS src_id, count(DISTINCT dst) AS reach_cnt
            RETURN min(reach_cnt) AS min_reach, max(reach_cnt) AS max_reach,
                   avg(reach_cnt) AS avg_reach,
                   percentileCont(toFloat(reach_cnt), 0.5) AS median_reach,
                   count(src_id) AS active_node_cnt
        """)

        self._query('extreme-1', 'Multi-dim aggregation: 3-dimensional GROUP BY (grp, status, kind)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)-[:BELONGS_TO]->(g:Grp)
            WHERE l.val IS NOT NULL AND g.depth = 1
            RETURN g.grp_id AS grp, n.status AS status, l.kind AS kind,
                   count(l) AS cnt, avg(l.val) AS avg_val,
                   percentileCont(l.val, 0.9) AS p90_val
            ORDER BY grp, status, kind
        """)

        self._query('extreme-2', 'Triangle count per node (local clustering coefficient numerator)', lambda s: f"""
            MATCH (a:Node)-[:LINKED]->(b:Node)-[:LINKED]->(c:Node)-[:LINKED]->(a)
            WITH a, count(DISTINCT [b.node_id, c.node_id]) AS triangle_count
            RETURN a.node_id, a.tag, triangle_count
            ORDER BY triangle_count DESC
            LIMIT 20
        """)

        self._query('extreme-3', 'PageRank-style: iterative in-degree weighted by source out-degree', lambda s: f"""
            MATCH (n:Node)
            WITH collect(n.node_id) AS all_ids, count(n) AS n_total
            MATCH (src:Node)-[:LINKED]->(dst:Node)
            WITH dst, src, n_total
            OPTIONAL MATCH (src)-[:LINKED]->(out:Node)
            WITH dst, src, count(out) AS src_out_deg, n_total
            WITH dst, sum(1.0 / (src_out_deg + 1)) AS pr_score
            RETURN dst.node_id, dst.tag, pr_score
            ORDER BY pr_score DESC
            LIMIT 20
        """)

        self._query('extreme-4', 'Multi-hop aggregation with intermediate rollup at each hop level', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[:LINKED*1..4]->(dst:Node)
            WITH dst, min(length(shortestPath((src)-[:LINKED*1..4]->(dst)))) AS hops
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(dst)
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(dst)
            WITH hops, count(DISTINCT dst) AS node_cnt,
                 avg(l.val) AS avg_log_val, avg(m.val) AS avg_measure_val
            RETURN hops, node_cnt, avg_log_val, avg_measure_val
            ORDER BY hops
        """)

        self._query('extreme-5', 'Full hierarchy stats: rollup from leaf nodes to root', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})<-[:CHILD_OF*0..5]-(g:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n) WHERE l.val IS NOT NULL
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n) WHERE m.dim = {s._param_dim()}
            OPTIONAL MATCH (n)-[:LINKED]->(nb:Node)
            WITH root, g.depth AS level,
                 count(DISTINCT n) AS node_cnt,
                 sum(CASE WHEN n.is_active THEN 1 ELSE 0 END) AS active_cnt,
                 count(l) AS log_cnt, avg(l.val) AS avg_log,
                 count(m) AS measure_cnt, avg(m.val) AS avg_measure,
                 count(nb) AS total_links
            RETURN root.grp_id, root.name, level,
                   node_cnt, active_cnt,
                   log_cnt, avg_log,
                   measure_cnt, avg_measure, total_links
            ORDER BY level, node_cnt DESC
        """)

        self._query('extreme-6', 'Connected component approximation via BFS depth-first CALL', lambda s: f"""
            MATCH (seed:Node {{node_id: {s._param_node_id()}}})
            CALL (seed) {{
                MATCH (seed)-[:LINKED*0..5]->(member:Node)
                RETURN DISTINCT member
            }}
            WITH collect(DISTINCT member.node_id) AS component
            RETURN size(component) AS component_size,
                   component[0..10] AS sample_member_ids
        """)

        self._query('extreme-7', 'Cross-product style 3-way group comparison', lambda s: f"""
            MATCH (g1:Grp {{depth: 1}})
            MATCH (g2:Grp {{depth: 1}})
            WHERE g1.grp_id < g2.grp_id
            OPTIONAL MATCH (n1:Node)-[:BELONGS_TO]->(g1) WHERE n1.is_active = true
            OPTIONAL MATCH (n2:Node)-[:BELONGS_TO]->(g2) WHERE n2.is_active = true
            WITH g1, g2, count(DISTINCT n1) AS active1, count(DISTINCT n2) AS active2
            RETURN g1.grp_id, g1.name, active1,
                   g2.grp_id, g2.name, active2,
                   abs(active1 - active2) AS imbalance
            ORDER BY imbalance DESC
            LIMIT 20
        """)

        self._query('extreme-8', 'Temporal decay: log val weighted by recency (exponential-like)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()} AND l.val IS NOT NULL
            WITH n, l,
                 duration.between(datetime(l.occurred_at), datetime()).days AS age_days
            WITH n,
                 sum(l.val * exp(-0.05 * toFloat(age_days))) AS decay_score,
                 count(l) AS log_cnt
            RETURN n.node_id, n.tag, log_cnt, decay_score
            ORDER BY decay_score DESC
            LIMIT 20
        """)

        self._query('extreme-9', 'Monster 8-entity: hierarchy+network+all entity types+stats', lambda s: f"""
            MATCH (root:Grp {{depth: 0}})<-[:CHILD_OF]-(g:Grp)
            MATCH (n:Node)-[:BELONGS_TO]->(g)
            WHERE n.is_active = true
            OPTIONAL MATCH (l:Log)-[:EVENT_OF]->(n) WHERE l.val IS NOT NULL
            OPTIONAL MATCH (m:Measure)-[:METRIC_OF]->(n) WHERE m.dim = {s._param_dim()}
            OPTIONAL MATCH (d:Doc)-[:DOC_OF]->(n)
            OPTIONAL MATCH (n)-[r:LINKED]->(nb:Node)
            WITH root, g, n,
                 count(l) AS log_cnt, avg(l.val) AS avg_log,
                 count(m) AS measure_cnt, avg(m.val) AS avg_measure,
                 count(d) AS doc_cnt, count(r) AS out_deg,
                 sum(r.weight) AS total_weight
            WITH root, g,
                 count(n) AS node_cnt,
                 sum(log_cnt) AS total_logs,
                 avg(avg_log) AS meta_avg_log,
                 sum(measure_cnt) AS total_measures,
                 avg(avg_measure) AS meta_avg_measure,
                 sum(doc_cnt) AS total_docs,
                 avg(out_deg) AS avg_out_deg,
                 sum(total_weight) AS sum_weights
            RETURN root.grp_id, root.name, g.grp_id, g.name,
                   node_cnt, total_logs, meta_avg_log,
                   total_measures, meta_avg_measure,
                   total_docs, avg_out_deg, sum_weights
            ORDER BY total_logs DESC
            LIMIT 20
        """)

        #endregion
        #region FOREACH-style batch writes via UNWIND

        self._query('foreach-0', 'UNWIND+CREATE batch Log inserts for a node', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CALL (n) {{
                MATCH (ex:Log) RETURN max(ex.log_id) AS base_id
            }}
            WITH n, base_id
            UNWIND range(1, {s._param_int('batch', 5, 20)}) AS i
            CREATE (l:Log {{
                log_id:      base_id + i,
                node_id:     n.node_id,
                kind:        i % 8,
                val:         toFloat(i) * 2.5,
                occurred_at: datetime() - duration({{seconds: i * 60}})
            }})
            CREATE (l)-[:EVENT_OF]->(n)
            RETURN count(l) AS created
        """)

        self._query('foreach-1', 'UNWIND+MERGE batch upsert Measure rows per dim', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CALL (n) {{
                MATCH (ex:Measure) RETURN max(ex.measure_id) AS base_id
            }}
            WITH n, base_id
            UNWIND [0, 1, 2, 3, 4] AS dim
            MERGE (m:Measure {{node_id: n.node_id, dim: dim}})
            ON CREATE SET m.measure_id = base_id + dim + 1,
                          m.val = toFloat(dim) * 10.0 + {s._param_float('offset', 0.1, 5.0, 2)},
                          m.recorded_at = date()
            ON MATCH  SET m.val = m.val * 1.01
            RETURN count(m) AS upserted
        """)

        self._query('foreach-2', 'UNWIND list of node_ids + SET property in batch', lambda s: f"""
            WITH [{s._param_node_ids(5, 15)}] AS ids
            UNWIND ids AS nid
            MATCH (n:Node {{node_id: nid}})
            SET n.val_int = n.val_int + {s._param_int('increment', 1, 10)}
            RETURN count(n) AS updated
        """)

        self._query('foreach-3', 'UNWIND edges list + CREATE LINKED relationships in batch', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})
            WITH src, [{s._param_node_ids(3, 8)}] AS dst_ids
            CALL (src) {{
                MATCH (ex:Log) RETURN count(ex) AS dummy
            }}
            UNWIND dst_ids AS did
            MATCH (dst:Node {{node_id: did}})
            WHERE src <> dst
              AND NOT EXISTS {{ MATCH (src)-[:LINKED]->(dst) }}
            CREATE (src)-[:LINKED {{
                kind: {s._param_link_kind()},
                weight: {s._param_float('w', 0.1, 0.9, 3)},
                created_at: datetime()
            }}]->(dst)
            RETURN count(*) AS created
        """)

        self._query('foreach-4', 'UNWIND + DETACH DELETE batch Log cleanup per kind', lambda s: f"""
            WITH [0, 1, 2] AS old_kinds
            UNWIND old_kinds AS k
            MATCH (l:Log)
            WHERE l.kind = k
              AND l.occurred_at < datetime('{s._param_date_minus_days(365, 730)}')
            WITH l LIMIT 20
            DETACH DELETE l
            RETURN count(*) AS deleted
        """)

        #endregion

    def _register_write_queries(self):

        #region INSERT / CREATE

        self._write_query('ins-0', 'CREATE single Log node + EVENT_OF rel (seed id)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CREATE (l:Log {{
                log_id:      {s._param_seed('log')},
                node_id:     n.node_id,
                kind:        {s._param_log_kind()},
                val:         {s._param_val()},
                occurred_at: datetime()
            }})
            CREATE (l)-[:EVENT_OF]->(n)
            RETURN l.log_id
        """)

        self._write_query('ins-1', 'CREATE multiple Log nodes via UNWIND over range', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CALL (n) {{
                MATCH (existing:Log)
                RETURN max(existing.log_id) AS base_id
            }}
            WITH n, base_id
            UNWIND range(1, {s._param_int('n_rows', 3, 10)}) AS i
            CREATE (l:Log {{
                log_id:      base_id + i,
                node_id:     n.node_id,
                kind:        (i % 8),
                val:         toFloat(i) * 1.5,
                occurred_at: datetime() - duration({{days: i}})
            }})
            CREATE (l)-[:EVENT_OF]->(n)
            RETURN count(l) AS created
        """)

        self._write_query('ins-2', 'CREATE Log rows from existing Log via MATCH + CREATE (copy with rescale)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND l.val IS NOT NULL
            WITH l, n
            LIMIT 5
            CALL (l, n) {{
                MATCH (existing:Log)
                RETURN max(existing.log_id) AS base_id
            }}
            CREATE (new_l:Log {{
                log_id:      base_id + l.log_id,
                node_id:     n.node_id,
                kind:        l.kind,
                val:         l.val * 0.9,
                occurred_at: datetime()
            }})
            CREATE (new_l)-[:EVENT_OF]->(n)
            RETURN count(new_l) AS created
        """)

        self._write_query('ins-3', 'CREATE Measure node (seed id) via index-filtered source', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CREATE (m:Measure {{
                measure_id:  {s._param_seed('measure')},
                node_id:     n.node_id,
                dim:         {s._param_dim()},
                val:         {s._param_val()},
                recorded_at: date()
            }})
            CREATE (m)-[:METRIC_OF]->(n)
            RETURN m.measure_id
        """)

        self._write_query('ins-4', 'CREATE LINKED relationship between two nodes (conditional)', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}}),
                  (dst:Node {{node_id: {s._param_node_id('dst_id')}}})
            WHERE src <> dst
              AND NOT EXISTS {{ MATCH (src)-[:LINKED]->(dst) }}
            CREATE (src)-[:LINKED {{
                kind:       {s._param_link_kind()},
                weight:     {s._param_float('weight', 0.01, 0.99, 4)},
                created_at: datetime()
            }}]->(dst)
            RETURN count(*) AS created
        """)

        self._write_query('ins-5', 'CREATE Doc node + DOC_OF rel (no existing doc on node)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            WHERE NOT EXISTS {{ MATCH (:Doc)-[:DOC_OF]->(n) }}
            CALL (n) {{
                MATCH (existing:Doc)
                RETURN max(existing.doc_id) AS max_id
            }}
            CREATE (d:Doc {{
                doc_id:     max_id + 1,
                node_id:    n.node_id,
                body:       'generated body text',
                meta:       '{{\"lang\":\"en_US\",\"version\":1,\"tags\":[],\"score\":5.0}}',
                created_at: datetime()
            }})
            CREATE (d)-[:DOC_OF]->(n)
            RETURN d.doc_id
        """)

        self._write_query('ins-6', 'MERGE Grp (upsert pattern, no duplicate on grp_id)', lambda s: f"""
            MERGE (g:Grp {{grp_id: {s._param_grp_id()}}})
            ON CREATE SET g.name = 'created_grp', g.depth = 1, g.priority = 0.5
            ON MATCH  SET g.priority = g.priority + 0.01
            RETURN g.grp_id, g.name, g.priority
        """)

        self._write_query('ins-7', 'MERGE LINKED relationship (upsert edge)', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}}),
                  (dst:Node {{node_id: {s._param_node_id('dst_id')}}})
            WHERE src <> dst
            MERGE (src)-[r:LINKED {{kind: {s._param_link_kind()}}}]->(dst)
            ON CREATE SET r.weight = {s._param_float('weight', 0.1, 0.9, 4)}, r.created_at = datetime()
            ON MATCH  SET r.weight = r.weight * 1.05
            RETURN r.weight
        """)

        self._write_query('ins-8', 'CREATE Log (seed id) + RETURN (ModifyTable with RETURNING equivalent)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CREATE (l:Log {{
                log_id:      {s._param_seed('log')},
                node_id:     n.node_id,
                kind:        {s._param_log_kind()},
                val:         null,
                occurred_at: datetime()
            }})
            CREATE (l)-[:EVENT_OF]->(n)
            RETURN l.log_id, l.kind, l.occurred_at
        """)

        self._write_query('ins-9', 'MERGE Node then CREATE Log in one chained query', lambda s: f"""
            MERGE (n:Node {{node_id: {s._param_node_id()}}})
            WITH n
            CALL (n) {{
                MATCH (existing:Log)
                RETURN max(existing.log_id) AS max_id
            }}
            CREATE (l:Log {{
                log_id:      max_id + 1,
                node_id:     n.node_id,
                kind:        {s._param_log_kind()},
                val:         {s._param_val()},
                occurred_at: datetime()
            }})
            CREATE (l)-[:EVENT_OF]->(n)
            RETURN l.log_id
        """)

        #endregion
        #region UPDATE / SET

        self._write_query('upd-0', 'SET single Node property by node_id (PK index)', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            SET n.val_int = {s._param_val_int()}
            RETURN n.node_id, n.val_int
        """)

        self._write_query('upd-1', 'SET Log.val for rows filtered by node_id + kind', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.node_id = {s._param_node_id()}
              AND l.kind = {s._param_log_kind()}
            SET l.val = coalesce(l.val, 0.0) + {s._param_float('val_increment', 0.1, 5.0, 4)}
            RETURN count(l) AS updated
        """)

        self._write_query('upd-2', 'SET status on many Nodes by non-indexed filter (seq scan)', lambda s: f"""
            MATCH (n:Node)
            WHERE n.val_int < {s._param_int('threshold', 50, 150)}
              AND n.is_active = false
            SET n.status = 1
            RETURN count(n) AS updated
        """)

        self._write_query('upd-3', 'SET with correlated subquery: update val_int to avg measure val', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})
            CALL (n) {{
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN avg(m.val) AS avg_m
            }}
            SET n.val_float = avg_m
            RETURN n.node_id, n.val_float
        """)

        self._write_query('upd-4', 'SET from aggregated MATCH (join-based update)', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WITH n, g
            CALL (n) {{
                MATCH (m:Measure)-[:METRIC_OF]->(n)
                RETURN count(m) AS m_cnt
            }}
            SET n.val_int = CASE WHEN m_cnt > 0 THEN m_cnt ELSE n.val_int END
            RETURN count(n) AS updated
        """)

        self._write_query('upd-5', 'SET on nodes in a group matching a subquery condition', lambda s: f"""
            MATCH (n:Node)-[:BELONGS_TO]->(g:Grp {{grp_id: {s._param_grp_id()}}})
            WHERE EXISTS {{
                MATCH (l:Log)-[:EVENT_OF]->(n)
                WHERE l.val IS NOT NULL AND l.val > {s._param_float('hi', 70.0, 95.0)}
            }}
            SET n.status = 2
            RETURN count(n) AS updated
        """)

        self._write_query('upd-6', 'SET LINKED weight on edges of high-degree node', lambda s: f"""
            MATCH (n:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(dst:Node)
            WHERE r.kind = {s._param_link_kind()}
            SET r.weight = r.weight * {s._param_float('weight_multiplier', 0.8, 1.2, 4)}
            RETURN count(r) AS updated
        """)

        self._write_query('upd-7', 'SET large range of Measure vals (high-volume update)', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)
            WHERE m.dim = {s._param_dim()}
              AND m.recorded_at >= date('{s._param_date_minus_days(1, 30)}')
            SET m.val = m.val * 1.01
            RETURN count(m) AS updated
        """)

        self._write_query('upd-8', 'SET with EXISTS correlated check in WHERE', lambda s: f"""
            MATCH (n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND NOT EXISTS {{ MATCH (d:Doc)-[:DOC_OF]->(n) }}
            SET n.is_active = false
            RETURN count(n) AS updated
        """)

        self._write_query('upd-9', 'SET multiple properties + RETURN (RETURNING equivalent)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.log_id = {s._param_log_id()}
            SET l.val = coalesce(l.val, 0.0) + {s._param_float('val_increment', 0.1, 10.0, 4)},
                l.kind = {s._param_log_kind()}
            RETURN l.log_id, l.val, l.kind, l.occurred_at
        """)

        #endregion
        #region DELETE / DETACH DELETE

        self._write_query('del-0', 'DELETE single Log by log_id (PK index)', lambda s: f"""
            MATCH (l:Log {{log_id: {s._param_log_id()}}})
            DELETE l
            RETURN count(*) AS deleted
        """)

        self._write_query('del-1', 'DELETE Log nodes by occurred_at range (index on occurred_at)', lambda s: f"""
            MATCH (l:Log)
            WHERE l.occurred_at < datetime('{s._param_date_minus_days(300, 700)}')
              AND l.kind = {s._param_log_kind()}
            DETACH DELETE l
            RETURN count(*) AS deleted
        """)

        self._write_query('del-2', 'DELETE Measure nodes by non-indexed val threshold', lambda s: f"""
            MATCH (m:Measure)
            WHERE m.val < {s._param_float('lo', 0.1, 5.0)}
              AND m.dim = {s._param_dim()}
            DETACH DELETE m
            RETURN count(*) AS deleted
        """)

        self._write_query('del-3', 'DELETE Log rows via IN subquery (semi-join pattern)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.node_id IN [{s._param_node_ids(5, 10)}]
              AND l.val IS NULL
            DETACH DELETE l
            RETURN count(*) AS deleted
        """)

        self._write_query('del-4', 'DELETE Log with USING-style join on Node property', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node)
            WHERE n.grp_id = {s._param_grp_id()}
              AND n.status = 4
              AND l.occurred_at < datetime('{s._param_date_minus_days(180, 730)}')
            DETACH DELETE l
            RETURN count(*) AS deleted
        """)

        self._write_query('del-5', 'DELETE with EXISTS correlated subquery', lambda s: f"""
            MATCH (m:Measure)-[:METRIC_OF]->(n:Node)
            WHERE NOT EXISTS {{ MATCH (n)-[:LINKED]->(:Node) }}
              AND m.dim = {s._param_dim()}
              AND m.val < {s._param_float('lo', 1.0, 10.0)}
            DETACH DELETE m
            RETURN count(*) AS deleted
        """)

        self._write_query('del-6', 'DELETE LINKED relationship (not node) on kind filter', lambda s: f"""
            MATCH (src:Node {{node_id: {s._param_node_id()}}})-[r:LINKED]->(dst:Node)
            WHERE r.kind = {s._param_link_kind()}
              AND r.weight < {s._param_weight()}
            DELETE r
            RETURN count(r) AS deleted
        """)

        self._write_query('del-7', 'DELETE oldest N Log rows for a node (bounded delete)', lambda s: f"""
            MATCH (l:Log)-[:EVENT_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WITH l ORDER BY l.occurred_at ASC
            LIMIT {s._param_int('batch_size', 5, 20)}
            DETACH DELETE l
            RETURN count(*) AS deleted
        """)

        self._write_query('del-8', 'DELETE Doc + DOC_OF rel via DETACH DELETE + RETURN', lambda s: f"""
            MATCH (d:Doc)-[:DOC_OF]->(n:Node {{node_id: {s._param_node_id()}}})
            WITH d
            LIMIT 1
            DETACH DELETE d
            RETURN count(*) AS deleted
        """)

        self._write_query('del-9', 'Batched DELETE via CALL subquery with LIMIT', lambda s: f"""
            CALL () {{
                MATCH (l:Log)
                WHERE l.val IS NULL
                  AND l.kind = {s._param_log_kind()}
                WITH l LIMIT {s._param_int('batch_size', 10, 50)}
                DETACH DELETE l
                RETURN count(*) AS batch_deleted
            }}
            RETURN batch_deleted
        """)

        #endregion
