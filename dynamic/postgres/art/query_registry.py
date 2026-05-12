from typing_extensions import Self, override
from core.drivers import DriverType
from ...common.art.query_registry import ArtQueryRegistry

def export():
    return PostgresArtQueryRegistry()

class PostgresArtQueryRegistry(ArtQueryRegistry[str]):
    """PostgreSQL query registry for the ART schema."""

    def __init__(self):
        super().__init__(DriverType.POSTGRES)

    @override
    def _register_queries(self):
        self._register_read_queries()
        self._register_write_queries()

    def _register_read_queries(self):
        # Coverage map
        # ============
        # pk-0       ... pk-1          Point lookups (PK; PK + wide JOIN)
        # scan-0     ... scan-20       Single-table scans & filters
        #                             Indexed TEXT, non-indexed INT exact/range,
        #                             TIMESTAMPTZ range, low-cardinality status,
        #                             partial index, AND/OR, LIKE, IS NULL/NOT NULL,
        #                             IN (small/large), NOT IN, ORDER BY + LIMIT,
        #                             DISTINCT, COUNT(*)
        # agg-0      ... agg-6         Single-table aggregations
        #                             SUM/AVG/MIN/MAX, GROUP BY low-/high-cardinality,
        #                             GROUP BY multi-column, HAVING, top-N, FILTER
        # join-0     ... join-13       Joins
        #                             INNER N:1 / 1:N, LEFT JOIN, anti-join, filter
        #                             on both sides, JOIN + aggregate, 3-way / 4-way
        #                             / star join, self-join 1-hop / 2-hop, FULL
        #                             OUTER JOIN, range-condition join, derived table
        # subquery-0 ... subquery-7    Subqueries
        #                             Scalar in SELECT / WHERE, IN / NOT IN /
        #                             EXISTS / NOT EXISTS, correlated in SELECT / WHERE
        # cte-0      ... cte-4         CTEs
        #                             Single CTE, two sequential CTEs, CTE -> CTE,
        #                             recursive ancestor path, recursive subtree + agg
        # window-0   ... window-5      Window functions
        #                             ROW_NUMBER, RANK (partitioned), running SUM,
        #                             LAG, AVG OVER PARTITION, NTILE
        # set-op-0   ... set-op-3      Set operations  UNION ALL / UNION / INTERSECT / EXCEPT
        # jsonb-0    ... jsonb-3       JSON  field access, filter, json_build_object, jsonb_agg
        # lateral-0               LATERAL join (top-K per group)
        # adv-0      ... adv-8         Advanced single-feature
        #                             CASE in SELECT/ORDER BY, DISTINCT ON, pagination
        #                             LIMIT+OFFSET, complex multi-CTE+window+join,
        #                             full-table large output, wide-row, BETWEEN on
        #                             large table, many-column FILTER aggregate, ROLLUP
        # graph-0    ... graph-8       Graph / multi-way joins
        #                             5-way / 6-way / 8-source joins, multi-dim pivot,
        #                             3-hop reachability, bidirectional neighbors,
        #                             common-neighbor INTERSECT, triangle detection,
        #                             hub in/out-degree analysis
        # window-adv-0 ... window-adv-4  Advanced window functions
        #                             Named WINDOW clause, FIRST/LAST_VALUE + frames,
        #                             PERCENTILE_CONT/DISC, 3-frame comparison, Pareto
        # cte-adv-0  ... cte-adv-3    Advanced CTEs
        #                             4-CTE pipeline + RANK, CTE + LATERAL, recursive
        #                             path accumulation, two independent recursive CTEs
        # nested-0   ... nested-3     Deeply nested subqueries
        #                             3-level nesting, ALL / ANY, HAVING subqueries,
        #                             5 correlated scalars in SELECT
        # pivot-0    ... pivot-2      Pivot / for-loop aggregates
        #                             Log-kind FILTER pivot, per-dim UNION ALL,
        #                             CASE double pivot
        # set-op-4                INTERSECT chain across 3 tables
        # rollup-0   ... rollup-2     Grouping extensions  CUBE / GROUPING SETS / ROLLUP
        # misc-0     ... misc-9       Misc special features
        #                             TABLESAMPLE, string_agg, array_agg+unnest,
        #                             regex ~*, generate_series histogram, COALESCE,
        #                             GREATEST/LEAST, NULLIF, VALUES lookup table,
        #                             monster query (5 CTEs + windows + 5-way join)
        # complex-0  ... complex-10   Complex multi-feature queries
        #                             Per-kind graph stats, temporal correlation,
        #                             multi-col ORDER BY + LATERAL, CTE+RANK+LATERAL,
        #                             z-score cascading agg, neighbor pivot, compound
        #                             set ops, node self-join pairs, REGR_SLOPE,
        #                             DISTINCT ON latest state, full degree z-score
        # bitmap-0   ... bitmap-2     Bitmap scans  BitmapOr / BitmapAnd
        # idx-only-0 ... idx-only-2   Index-only scans
        # memoize-0  ... memoize-1    Memoize node
        # merge-join-0 ... merge-join-1  Merge join candidates
        # mat-cte-0  ... mat-cte-1   MATERIALIZED / NOT MATERIALIZED CTEs
        # parallel-0 ... parallel-2   Parallel execution  agg / hash join / seq scan
        # merge-append-0          MergeAppend (pre-sorted UNION ALL + ORDER BY)
        # lock-0     ... lock-2       LockRows  FOR UPDATE / FOR SHARE / SKIP LOCKED
        # jsonb-op-0 ... jsonb-op-3   JSONB operators  @> / ? / #>> / jsonb_path_exists
        # datetime-0 ... datetime-3   Date/time functional expressions
        #                             DATE_TRUNC group-by, EXTRACT(DOW), age(),
        #                             monthly aggregation
        # win-frame-0 ... win-frame-5  Window frame variants
        #                             GROUPS BETWEEN, RANGE BETWEEN interval,
        #                             CUME_DIST/PERCENT_RANK, DENSE_RANK/RANK/ROW_NUMBER,
        #                             NTH_VALUE, conditional cumulative SUM via CASE
        # distinct-0 ... distinct-2   DISTINCT variants  Sort / Hash / multi-column
        # incr-sort-0 ... incr-sort-1  Incremental sort candidates
        # selectivity-0 ... selectivity-2  Selectivity extremes (ultra-selective / near-full)
        # join-size-0 ... join-size-2  Join size classes  Small x Large / Large x Large / Medium x Large
        # plan-0     ... plan-9       Misc plan-operator patterns
        #                             Values Scan, CASE in GROUP BY, Subquery Scan,
        #                             EXCEPT ALL, INTERSECT ALL, generate_series x node,
        #                             subquery in ORDER BY, EXISTS+NOT EXISTS,
        #                             multi-level EXISTS, LATERAL with filter

        #region Point lookups

        self._query('pk-0', 'PK point lookup - narrow output', lambda s: f"""
            SELECT node_id, tag, val_int, val_float, status, grp_id, created_at, is_active
            FROM node
            WHERE node_id = {s._param_node_id()}
        """)

        self._query('pk-1', 'PK point lookup - wide output (node + doc body)', lambda s: f"""
            SELECT n.node_id, n.tag, n.note, d.body, d.meta
            FROM node n
            LEFT JOIN doc d ON d.node_id = n.node_id
            WHERE n.node_id = {s._param_node_id()}
        """)

        #endregion
        #region Single-table scans & filters

        self._query('scan-0', 'Exact filter on indexed TEXT column (tag)', lambda s: f"""
            SELECT node_id, tag, val_int, grp_id
            FROM node
            WHERE tag = '{s._param_tag()}'
        """)

        self._query('scan-1', 'Exact filter on non-indexed INTEGER column (val_int)', lambda s: f"""
            SELECT node_id, tag, val_int
            FROM node
            WHERE val_int = {s._param_val_int()}
        """)

        self._query('scan-2', 'Range filter on non-indexed INTEGER column (val_int BETWEEN)', lambda s: f"""
            SELECT node_id, tag, val_int
            FROM node
            WHERE val_int BETWEEN {s._param_int('val_lo', 1, 800)} AND {s._param_int('val_hi', 200, 1000)}
        """)

        self._query('scan-3', 'Half-open range filter on indexed TIMESTAMPTZ (created_at >=)', lambda s: f"""
            SELECT node_id, tag, created_at
            FROM node
            WHERE created_at >= '{s._param_date_minus_days(30, 365)}'
        """)

        self._query('scan-4', 'Closed range filter on indexed TIMESTAMPTZ (BETWEEN)', lambda s: f"""
            SELECT node_id, tag, created_at
            FROM node
            WHERE created_at BETWEEN '{s._param_date_minus_days(60, 180)}' AND '{s._param_date_minus_days(1,   59)}'
        """)

        self._query('scan-5', 'Filter on low-cardinality column without dedicated index (status)', lambda s: f"""
            SELECT node_id, tag, status
            FROM node
            WHERE status = {s._param_status()}
        """)

        self._query('scan-6', 'Partial-index usage (is_active = TRUE + grp_id)', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE is_active = TRUE
              AND grp_id = {s._param_grp_id()}
        """)

        self._query('scan-7', 'Multi-condition AND: indexed column + non-indexed column', lambda s: f"""
            SELECT node_id, tag, val_int, grp_id
            FROM node
            WHERE tag = '{s._param_tag()}'
              AND val_int > {s._param_val_int()}
        """)

        self._query('scan-8', 'Multi-condition AND: two non-indexed columns', lambda s: f"""
            SELECT node_id, val_int, val_float, status
            FROM node
            WHERE val_int > {s._param_int('val_lo', 400, 800)}
              AND status = {s._param_status()}
        """)

        self._query('scan-9', 'Multi-condition OR: both non-indexed (wide result expected)', lambda s: f"""
            SELECT node_id, tag, val_int, status
            FROM node
            WHERE val_int < {s._param_int('val_threshold', 100, 300)}
              AND status = {s._param_status()}
        """)

        self._query('scan-10', 'LIKE prefix search on unindexed TEXT column (note)', lambda s: f"""
            SELECT node_id, note
            FROM node
            WHERE note LIKE '{s._param('prefix', lambda: chr(ord('a') + s._rng_int(0, 25)))}%'
            LIMIT 200
        """)

        self._query('scan-11', 'IS NULL filter on nullable column (log.val)', lambda s: f"""
            SELECT log_id, node_id, kind, occurred_at
            FROM log
            WHERE val IS NULL
              AND kind = {s._param_log_kind()}
        """)

        self._query('scan-12', 'IS NOT NULL filter on nullable column (log.val)', lambda s: f"""
            SELECT log_id, node_id, val, occurred_at
            FROM log
            WHERE val IS NOT NULL
              AND node_id = {s._param_node_id()}
        """)

        self._query('scan-13', 'IN predicate with small array (5 grp_ids) - likely index scan', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE grp_id IN ({s._param_grp_ids(5, 5)})
        """)

        self._query('scan-14', 'IN predicate with large array (50 grp_ids) - likely hash/bitmap', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE grp_id IN ({s._param_grp_ids(50, 50)})
        """)

        self._query('scan-15', 'NOT IN predicate with active-node filter', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE grp_id NOT IN ({s._param_grp_ids(10, 20)})
              AND is_active = TRUE
        """)

        self._query('scan-16', 'ORDER BY indexed column + LIMIT (index scan + early stop)', lambda s: f"""
            SELECT node_id, tag, created_at
            FROM node
            WHERE is_active = TRUE
            ORDER BY created_at DESC
            LIMIT {s._param_limit(10, 4)}
        """)

        self._query('scan-17', 'ORDER BY non-indexed column + LIMIT (full sort required)', lambda s: f"""
            SELECT node_id, val_int, val_float
            FROM node
            WHERE status = {s._param_status()}
            ORDER BY val_float DESC
            LIMIT {s._param_limit(10, 4)}
        """)

        self._query('scan-18', 'DISTINCT on low-cardinality column', lambda s: f"""
            SELECT DISTINCT status
            FROM node
            WHERE grp_id = {s._param_grp_id()}
            ORDER BY status
        """)

        self._query('scan-19', 'COUNT(*) with composite filter', lambda s: f"""
            SELECT COUNT(*)
            FROM node
            WHERE tag = '{s._param_tag()}'
              AND is_active = TRUE
        """)

        self._query('scan-20', 'COUNT(DISTINCT col) per group', lambda s: f"""
            SELECT status, COUNT(DISTINCT grp_id) AS distinct_grps
            FROM node
            WHERE created_at >= '{s._param_date_minus_days(30, 365)}'
            GROUP BY status
            ORDER BY status
        """)

        #endregion
        #region Single-table aggregations

        self._query('agg-0', 'SUM + AVG + MIN + MAX on nullable column (log.val per node)', lambda s: f"""
            SELECT
                COUNT(*)        AS n,
                SUM(val)        AS total,
                AVG(val)        AS avg_val,
                MIN(val)        AS min_val,
                MAX(val)        AS max_val
            FROM log
            WHERE node_id = {s._param_node_id()}
              AND val IS NOT NULL
        """)

        self._query('agg-1', 'GROUP BY low-cardinality column (status, 5 groups)', lambda s: f"""
            SELECT status, COUNT(*) AS cnt, AVG(val_float) AS avg_f
            FROM node
            GROUP BY status
            ORDER BY status
        """)

        self._query('agg-2', 'GROUP BY high-cardinality column (grp_id, ~200 groups)', lambda s: f"""
            SELECT grp_id, COUNT(*) AS cnt, SUM(val_int) AS sum_val
            FROM node
            WHERE is_active = TRUE
            GROUP BY grp_id
            ORDER BY grp_id
        """)

        self._query('agg-3', 'GROUP BY multiple columns (grp_id + status)', lambda s: f"""
            SELECT grp_id, status, COUNT(*) AS cnt
            FROM node
            GROUP BY grp_id, status
            ORDER BY grp_id, status
        """)

        self._query('agg-4', 'GROUP BY + HAVING to filter groups', lambda s: f"""
            SELECT grp_id, COUNT(*) AS cnt, SUM(val_int) AS total
            FROM node
            WHERE is_active = TRUE
            GROUP BY grp_id
            HAVING COUNT(*) >= {s._param_int('min_cnt', 20, 100)}
            ORDER BY total DESC
        """)

        self._query('agg-5', 'GROUP BY + ORDER BY + LIMIT (top-N groups)', lambda s: f"""
            SELECT grp_id, COUNT(*) AS cnt, AVG(val_float) AS avg_f
            FROM node
            GROUP BY grp_id
            ORDER BY avg_f DESC
            LIMIT {s._param_limit(10, 3)}
        """)

        self._query('agg-6', 'Aggregate FILTER clause (conditional counts & avg within one GROUP BY)', lambda s: f"""
            SELECT
                node_id,
                COUNT(*)                                             AS total_logs,
                COUNT(*) FILTER (WHERE kind = {s._param_log_kind()}) AS target_kind_logs,
                AVG(val) FILTER (WHERE val IS NOT NULL)              AS avg_val
            FROM log
            WHERE occurred_at >= '{s._param_date_minus_days(7, 90)}'
            GROUP BY node_id
            HAVING COUNT(*) > {s._param_int('min_logs', 1, 5)}
            ORDER BY total_logs DESC
            LIMIT 50
        """)

        #endregion
        #region Joins

        self._query('join-0', 'INNER JOIN N:1 (node -> grp lookup)', lambda s: f"""
            SELECT n.node_id, n.tag, n.val_int, g.name AS grp_name, g.priority
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            WHERE n.tag = '{s._param_tag()}'
        """)

        self._query('join-1', 'INNER JOIN 1:N (node -> log, recent events)', lambda s: f"""
            SELECT n.node_id, n.tag, l.log_id, l.kind, l.val, l.occurred_at
            FROM node n
            JOIN log l ON l.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
              AND l.occurred_at >= '{s._param_date_minus_days(7, 30)}'
            ORDER BY l.occurred_at DESC
            LIMIT 200
        """)

        self._query('join-2', 'LEFT JOIN (node -> doc, some nodes have no doc)', lambda s: f"""
            SELECT n.node_id, n.tag, d.doc_id, d.created_at AS doc_created
            FROM node n
            LEFT JOIN doc d ON d.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
            ORDER BY n.node_id
        """)

        self._query('join-3', 'Anti-join via LEFT JOIN + NULL check (nodes with no doc)', lambda s: f"""
            SELECT n.node_id, n.tag, n.grp_id
            FROM node n
            LEFT JOIN doc d ON d.node_id = n.node_id
            WHERE d.doc_id IS NULL
              AND n.is_active = TRUE
            ORDER BY n.node_id
            LIMIT 500
        """)

        self._query('join-4', 'INNER JOIN with filter predicate on both sides', lambda s: f"""
            SELECT n.node_id, n.tag, l.log_id, l.val
            FROM node n
            JOIN log l ON l.node_id = n.node_id
            WHERE n.status = {s._param_status()}
              AND l.kind   = {s._param_log_kind()}
              AND l.val    > {s._param_float('val_min', 10.0, 90.0)}
            ORDER BY l.val DESC
            LIMIT 100
        """)

        self._query('join-5', 'INNER JOIN + GROUP BY aggregate (node + measure)', lambda s: f"""
            SELECT n.node_id, n.tag,
                   COUNT(m.measure_id) AS n_measures,
                   AVG(m.val)          AS avg_val
            FROM node n
            JOIN measure m ON m.node_id = n.node_id
            WHERE n.grp_id IN ({s._param_grp_ids(5, 20)})
              AND m.dim = {s._param_dim()}
            GROUP BY n.node_id, n.tag
            ORDER BY avg_val DESC
            LIMIT 100
        """)

        self._query('join-6', '3-way join (node + grp + log)', lambda s: f"""
            SELECT n.node_id, n.tag, g.name AS grp_name,
                   l.kind, l.val, l.occurred_at
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            JOIN log  l ON l.node_id = n.node_id
            WHERE g.depth = {s._param_depth()}
              AND l.occurred_at >= '{s._param_date_minus_days(7, 30)}'
              AND l.val IS NOT NULL
            ORDER BY l.occurred_at DESC
            LIMIT 200
        """)

        self._query('join-7', '4-way join (node + grp + log + measure)', lambda s: f"""
            SELECT n.node_id, n.tag, g.name,
                   COUNT(DISTINCT l.log_id)     AS log_cnt,
                   COUNT(DISTINCT m.measure_id) AS measure_cnt,
                   AVG(m.val)                   AS avg_measure
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            LEFT JOIN log l ON l.node_id = n.node_id
                           AND l.occurred_at >= '{s._param_date_minus_days(14, 60)}'
            LEFT JOIN measure m ON m.node_id = n.node_id
                               AND m.dim = {s._param_dim()}
            WHERE g.grp_id IN ({s._param_grp_ids(3, 10)})
            GROUP BY n.node_id, n.tag, g.name
            ORDER BY log_cnt DESC
            LIMIT 50
        """)

        self._query('join-8', 'Star join (node + grp + doc + aggregated log subquery)', lambda s: f"""
            SELECT n.node_id, n.tag, g.name AS grp,
                   d.created_at               AS doc_ts,
                   COALESCE(la.log_count, 0)  AS log_count
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            LEFT JOIN doc d ON d.node_id = n.node_id
            LEFT JOIN (
                SELECT node_id, COUNT(*) AS log_count
                FROM log
                WHERE occurred_at >= '{s._param_date_minus_days(30, 90)}'
                GROUP BY node_id
            ) la ON la.node_id = n.node_id
            WHERE n.grp_id    = {s._param_grp_id()}
              AND n.is_active = TRUE
            ORDER BY n.node_id
        """)

        self._query('join-9', 'Self-join via link (1-hop out-neighbors of a node)', lambda s: f"""
            SELECT lk.dst_id, dst.tag, lk.kind, lk.weight
            FROM link lk
            JOIN node dst ON dst.node_id = lk.dst_id
            WHERE lk.src_id = {s._param_node_id()}
            ORDER BY lk.weight DESC
        """)

        def q_self_join_2hop(s: Self):
            nid = s._param_node_id()
            return f"""
                SELECT DISTINCT lk2.dst_id, dst.tag
                FROM link lk1
                JOIN link lk2 ON lk2.src_id = lk1.dst_id
                JOIN node dst  ON dst.node_id = lk2.dst_id
                WHERE lk1.src_id = {nid}
                  AND lk2.dst_id <> {nid}
                ORDER BY lk2.dst_id
                LIMIT 200
            """
        self._query('join-10', 'Self-join via link (2-hop reachable nodes)', q_self_join_2hop)

        self._query('join-11', 'FULL OUTER JOIN (filtered node set + doc)', lambda s: f"""
            SELECT COALESCE(n.node_id, d.node_id) AS node_id,
                   n.tag, d.doc_id, d.created_at
            FROM (
                SELECT node_id, tag FROM node WHERE grp_id = {s._param_grp_id()}
            ) n
            FULL OUTER JOIN doc d ON d.node_id = n.node_id
            ORDER BY 1
            LIMIT 500
        """)

        self._query('join-12', 'JOIN with range condition on join attribute (link.weight > threshold)', lambda s: f"""
            SELECT n.node_id, n.tag, lk.dst_id, lk.weight
            FROM node n
            JOIN link lk ON lk.src_id = n.node_id
            WHERE n.grp_id   = {s._param_grp_id()}
              AND lk.weight  > {s._param_weight()}
            ORDER BY lk.weight DESC
            LIMIT 100
        """)

        self._query('join-13', 'JOIN with aggregated derived table (node + agg log inline view)', lambda s: f"""
            SELECT n.node_id, n.tag, la.total_val, la.event_count
            FROM node n
            JOIN (
                SELECT node_id,
                       SUM(val)   AS total_val,
                       COUNT(*)   AS event_count
                FROM   log
                WHERE  kind      = {s._param_log_kind()}
                  AND  val IS NOT NULL
                GROUP BY node_id
            ) la ON la.node_id = n.node_id
            WHERE n.status = {s._param_status()}
            ORDER BY la.total_val DESC
            LIMIT 100
        """)

        #endregion
        #region Subqueries

        self._query('subquery-0', 'Scalar correlated subquery in SELECT (count logs per node)', lambda s: f"""
            SELECT n.node_id, n.tag,
                   (SELECT COUNT(*) FROM log l WHERE l.node_id = n.node_id) AS log_cnt
            FROM node n
            WHERE n.grp_id    = {s._param_grp_id()}
              AND n.is_active = TRUE
            ORDER BY n.node_id
            LIMIT 100
        """)

        self._query('subquery-1', 'Scalar subquery in WHERE (compare against group average)', lambda s: f"""
            SELECT node_id, tag, val_int
            FROM node
            WHERE val_int > (
                SELECT AVG(val_int) FROM node WHERE status = {s._param_status()}
            )
              AND is_active = TRUE
            ORDER BY val_int DESC
            LIMIT 100
        """)

        self._query('subquery-2', 'IN with uncorrelated subquery (nodes in high-priority groups)', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE grp_id IN (
                SELECT grp_id FROM grp WHERE priority > {s._param_priority()}
            )
              AND is_active = TRUE
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('subquery-3', 'NOT IN with uncorrelated subquery (exclude root-group nodes)', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE grp_id NOT IN (
                SELECT grp_id FROM grp WHERE depth = 0
            )
              AND is_active = TRUE
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('subquery-4', 'EXISTS subquery (nodes with a qualifying log event)', lambda s: f"""
            SELECT n.node_id, n.tag, n.grp_id
            FROM node n
            WHERE EXISTS (
                SELECT 1 FROM log l
                WHERE l.node_id = n.node_id
                  AND l.kind    = {s._param_log_kind()}
                  AND l.val     > {s._param_float('val_thresh', 50.0, 90.0)}
            )
              AND n.is_active = TRUE
            ORDER BY n.node_id
            LIMIT 200
        """)

        self._query('subquery-5', 'NOT EXISTS subquery - anti-join (nodes with no outgoing links)', lambda s: f"""
            SELECT n.node_id, n.tag, n.status
            FROM node n
            WHERE NOT EXISTS (
                SELECT 1 FROM link lk WHERE lk.src_id = n.node_id
            )
            ORDER BY n.node_id
            LIMIT 200
        """)

        self._query('subquery-6', 'Correlated scalar subquery in SELECT (max measure val per node)', lambda s: f"""
            SELECT n.node_id, n.tag,
                   (SELECT MAX(m.val)
                    FROM   measure m
                    WHERE  m.node_id = n.node_id
                      AND  m.dim     = {s._param_dim()}) AS max_measure
            FROM node n
            WHERE n.grp_id = {s._param_grp_id()}
            ORDER BY n.node_id
            LIMIT 100
        """)

        self._query('subquery-7', 'Correlated subquery in WHERE (val_int > per-group average)', lambda s: f"""
            SELECT node_id, tag, grp_id, val_int
            FROM node n
            WHERE val_int > (
                SELECT AVG(val_int) FROM node n2 WHERE n2.grp_id = n.grp_id
            )
            ORDER BY node_id
            LIMIT 200
        """)

        #endregion
        #region CTEs

        self._query('cte-0', 'Single CTE + main query (active nodes in a tag, enriched with grp)', lambda s: f"""
            WITH active_nodes AS (
                SELECT node_id, tag, grp_id, val_int
                FROM   node
                WHERE  is_active = TRUE
                  AND  tag       = '{s._param_tag()}'
            )
            SELECT an.node_id, an.tag, g.name, an.val_int
            FROM   active_nodes an
            JOIN   grp g ON g.grp_id = an.grp_id
            ORDER BY an.val_int DESC
            LIMIT 100
        """)

        self._query('cte-1', 'Two independent CTEs joined in main query', lambda s: f"""
            WITH
            filtered_nodes AS (
                SELECT node_id, grp_id, val_int
                FROM   node
                WHERE  is_active = TRUE
                  AND  val_int   > {s._param_int('val_thr', 500, 900)}
            ),
            grp_avg AS (
                SELECT grp_id, AVG(priority) AS avg_priority
                FROM   grp
                GROUP BY grp_id
            )
            SELECT fn.node_id, fn.grp_id, fn.val_int, ga.avg_priority
            FROM   filtered_nodes fn
            JOIN   grp_avg ga ON ga.grp_id = fn.grp_id
            ORDER BY fn.val_int DESC
            LIMIT 100
        """)

        self._query('cte-2', 'CTE that references another CTE (two-stage enrichment)', lambda s: f"""
            WITH
            high_activity AS (
                SELECT node_id, COUNT(*) AS log_cnt
                FROM   log
                WHERE  occurred_at >= '{s._param_date_minus_days(30, 90)}'
                GROUP BY node_id
                HAVING COUNT(*) > {s._param_int('min_cnt', 5, 20)}
            ),
            enriched AS (
                SELECT ha.node_id, ha.log_cnt, n.tag, n.grp_id
                FROM   high_activity ha
                JOIN   node n ON n.node_id = ha.node_id
            )
            SELECT e.node_id, e.tag, g.name, e.log_cnt
            FROM   enriched e
            JOIN   grp g ON g.grp_id = e.grp_id
            ORDER BY e.log_cnt DESC
            LIMIT 50
        """)

        self._query('cte-3', 'Recursive CTE - ancestor path traversal in grp', lambda s: f"""
            WITH RECURSIVE ancestors AS (
                SELECT grp_id, name, parent_id, depth, 0 AS level
                FROM   grp
                WHERE  grp_id = {s._param_grp_id()}
                UNION ALL
                SELECT g.grp_id, g.name, g.parent_id, g.depth, a.level + 1
                FROM   grp g
                JOIN   ancestors a ON g.grp_id = a.parent_id
            )
            SELECT grp_id, name, depth, level
            FROM   ancestors
            ORDER BY level
        """)

        self._query('cte-4', 'Recursive CTE - subtree size aggregation', lambda s: f"""
            WITH RECURSIVE subtree AS (
                SELECT grp_id, name, parent_id
                FROM   grp
                WHERE  grp_id = {s._param_grp_id()}
                UNION ALL
                SELECT g.grp_id, g.name, g.parent_id
                FROM   grp g
                JOIN   subtree st ON g.parent_id = st.grp_id
            )
            SELECT st.grp_id, st.name,
                   COUNT(n.node_id) AS node_count
            FROM   subtree st
            LEFT JOIN node n ON n.grp_id = st.grp_id
            GROUP BY st.grp_id, st.name
            ORDER BY node_count DESC
        """)

        #endregion
        #region Window functions

        self._query('window-0', 'ROW_NUMBER() OVER (ORDER BY) - global ranking by val_int', lambda s: f"""
            SELECT node_id, tag, val_int, grp_id,
                   ROW_NUMBER() OVER (ORDER BY val_int DESC) AS rn
            FROM node
            WHERE is_active = TRUE
              AND grp_id    = {s._param_grp_id()}
            LIMIT 100
        """)

        self._query('window-1', 'RANK() OVER (PARTITION BY grp_id ORDER BY val_int) - per-group rank', lambda s: f"""
            SELECT node_id, tag, grp_id, val_int,
                   RANK() OVER (PARTITION BY grp_id ORDER BY val_int DESC) AS grp_rank
            FROM node
            WHERE is_active = TRUE
            ORDER BY grp_id, grp_rank
            LIMIT 200
        """)

        self._query('window-2', 'Running SUM() OVER (PARTITION BY node_id ORDER BY recorded_at)', lambda s: f"""
            SELECT node_id, recorded_at, val,
                   SUM(val) OVER (PARTITION BY node_id ORDER BY recorded_at) AS running_total
            FROM measure
            WHERE node_id = {s._param_node_id()}
              AND dim     = {s._param_dim()}
            ORDER BY recorded_at
        """)

        self._query('window-3', 'LAG() for row-to-row delta within a partition', lambda s: f"""
            SELECT log_id, node_id, occurred_at, val,
                   LAG(val) OVER (PARTITION BY node_id ORDER BY occurred_at) AS prev_val,
                   val - LAG(val) OVER (PARTITION BY node_id ORDER BY occurred_at) AS delta
            FROM log
            WHERE node_id = {s._param_node_id()}
              AND val IS NOT NULL
            ORDER BY occurred_at
        """)

        self._query('window-4', 'AVG() OVER (PARTITION BY grp_id) - deviation from group mean', lambda s: f"""
            SELECT node_id, tag, grp_id, val_int,
                   AVG(val_int) OVER (PARTITION BY grp_id)              AS grp_avg,
                   val_int - AVG(val_int) OVER (PARTITION BY grp_id)    AS deviation
            FROM node
            WHERE is_active = TRUE
              AND grp_id IN ({s._param_grp_ids(5, 15)})
            ORDER BY grp_id, deviation DESC
        """)

        self._query('window-5', 'NTILE(10) for decile bucketing', lambda s: f"""
            SELECT node_id, tag, val_int,
                   NTILE(10) OVER (ORDER BY val_int) AS decile
            FROM node
            WHERE status = {s._param_status()}
            ORDER BY decile, val_int
            LIMIT 500
        """)

        #endregion
        #region Set operations

        self._query('set-op-0', 'UNION ALL - two disjoint range filters on the same table', lambda s: f"""
            SELECT node_id, tag, val_int, 'low'  AS bucket
            FROM   node
            WHERE  val_int < {s._param_int('lo_thr', 100, 300)}
              AND  is_active = TRUE
            UNION ALL
            SELECT node_id, tag, val_int, 'high' AS bucket
            FROM   node
            WHERE  val_int > {s._param_int('hi_thr', 700, 900)}
              AND  is_active = TRUE
            ORDER BY val_int
            LIMIT 500
        """)

        self._query('set-op-1', 'UNION (distinct) - nodes reachable from two different filters', lambda s: f"""
            SELECT node_id FROM node WHERE grp_id = {s._param_grp_id()}
            UNION
            SELECT src_id AS node_id FROM link WHERE kind = {s._param_link_kind()}
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('set-op-2', 'INTERSECT - nodes that satisfy two independent predicates', lambda s: f"""
            SELECT node_id FROM node WHERE tag = '{s._param_tag()}'
            INTERSECT
            SELECT node_id FROM node WHERE grp_id = {s._param_grp_id()} AND is_active = TRUE
            ORDER BY node_id
        """)

        self._query('set-op-3', 'EXCEPT - nodes in a group but not matching a tag', lambda s: f"""
            SELECT node_id FROM node WHERE grp_id = {s._param_grp_id()}
            EXCEPT
            SELECT node_id FROM node WHERE tag = '{s._param_tag()}'
            ORDER BY node_id
            LIMIT 200
        """)

        #endregion
        #region JSON operations

        self._query('jsonb-0', 'JSON field access with type cast in SELECT', lambda s: f"""
            SELECT doc_id, node_id,
                   meta->>'lang'               AS lang,
                   (meta->>'version')::INT     AS version,
                   (meta->>'score')::FLOAT8    AS score
            FROM doc
            WHERE node_id = {s._param_node_id()}
        """)

        self._query('jsonb-1', 'Filter on JSONB field value', lambda s: f"""
            SELECT doc_id, node_id, meta
            FROM doc
            WHERE (meta->>'version')::INT > {s._param_int('min_ver', 1, 3)}
              AND meta->>'lang' IS NOT NULL
            ORDER BY doc_id
            LIMIT 100
        """)

        self._query('jsonb-2', 'json_build_object in SELECT to construct structured output', lambda s: f"""
            SELECT n.node_id,
                   json_build_object(
                       'tag',     n.tag,
                       'val_int', n.val_int,
                       'grp',     g.name,
                       'active',  n.is_active
                   ) AS node_json
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            WHERE n.grp_id = {s._param_grp_id()}
            LIMIT 100
        """)

        self._query('jsonb-3', 'jsonb_agg - aggregate child documents per node into JSON array', lambda s: f"""
            SELECT n.node_id, n.tag,
                   jsonb_agg(
                       jsonb_build_object(
                           'doc_id',     d.doc_id,
                           'created_at', d.created_at,
                           'lang',       d.meta->>'lang'
                       ) ORDER BY d.created_at DESC
                   ) AS docs
            FROM node n
            JOIN doc d ON d.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
            GROUP BY n.node_id, n.tag
            ORDER BY n.node_id
            LIMIT 50
        """)

        #endregion
        #region Advanced patterns

        self._query('lateral-0', 'LATERAL join - top-3 logs per node (top-K per group)', lambda s: f"""
            SELECT n.node_id, n.tag,
                   top_l.log_id, top_l.val, top_l.occurred_at
            FROM node n
            CROSS JOIN LATERAL (
                SELECT log_id, val, occurred_at
                FROM   log
                WHERE  node_id  = n.node_id
                  AND  val IS NOT NULL
                ORDER BY val DESC
                LIMIT 3
            ) top_l
            WHERE n.grp_id    = {s._param_grp_id()}
              AND n.is_active = TRUE
            ORDER BY n.node_id, top_l.val DESC
            LIMIT 200
        """)

        self._query('adv-0', 'CASE expression in SELECT and ORDER BY', lambda s: f"""
            SELECT node_id, tag, val_int,
                   CASE
                       WHEN val_int < 250  THEN 'low'
                       WHEN val_int < 750  THEN 'mid'
                       ELSE                     'high'
                   END AS bucket,
                   CASE status
                       WHEN 0 THEN 'active'
                       WHEN 1 THEN 'inactive'
                       WHEN 2 THEN 'pending'
                       ELSE        'other'
                   END AS status_label
            FROM node
            WHERE grp_id = {s._param_grp_id()}
            ORDER BY bucket, val_int
            LIMIT 200
        """)

        self._query('adv-1', 'DISTINCT ON - latest log event per node (deduplication)', lambda s: f"""
            SELECT DISTINCT ON (node_id)
                node_id, log_id, kind, val, occurred_at
            FROM log
            WHERE occurred_at >= '{s._param_date_minus_days(7, 60)}'
            ORDER BY node_id, occurred_at DESC
            LIMIT 500
        """)

        self._query('adv-2', 'Pagination - LIMIT + OFFSET for keyset-style paging', lambda s: f"""
            SELECT node_id, tag, val_int, grp_id, created_at
            FROM node
            WHERE is_active = TRUE
            ORDER BY created_at DESC
            LIMIT  {s._param_limit(20, 3)}
            OFFSET {s._param_int('page_offset', 0, 1000)}
        """)

        self._query('adv-3', 'Complex: multi-CTE + RANK() window + JOIN (activity leaders per group)', lambda s: f"""
            WITH
            recent_logs AS (
                SELECT node_id,
                        COUNT(*)   AS log_cnt,
                        AVG(val)   AS avg_val
                FROM   log
                WHERE  occurred_at >= '{s._param_date_minus_days(7, 30)}'
                    AND  val IS NOT NULL
                GROUP BY node_id
            ),
            ranked_nodes AS (
                SELECT n.node_id, n.tag, n.grp_id,
                        rl.log_cnt, rl.avg_val,
                        RANK() OVER (
                            PARTITION BY n.grp_id
                            ORDER BY rl.log_cnt DESC
                        ) AS grp_rank
                FROM   node n
                JOIN   recent_logs rl ON rl.node_id = n.node_id
                WHERE  n.is_active = TRUE
            )
            SELECT rn.node_id, rn.tag, g.name AS grp,
                    rn.log_cnt, rn.avg_val, rn.grp_rank
            FROM   ranked_nodes rn
            JOIN   grp g ON g.grp_id = rn.grp_id
            WHERE  rn.grp_rank <= {s._param_int('top_k', 3, 10)}
            ORDER BY rn.grp_id, rn.grp_rank
            LIMIT 200
        """)

        self._query('adv-4', 'Large output - full-table scan with no row limit', lambda s: f"""
            SELECT node_id, tag, val_int, val_float, status, grp_id, created_at
            FROM node
            ORDER BY node_id
        """)

        self._query('adv-5', 'Wide-row output - few rows with large doc body', lambda s: f"""
            SELECT n.node_id, n.tag, n.note, d.body, d.meta
            FROM node n
            JOIN doc d ON d.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
            LIMIT 10
        """)

        self._query('adv-6', 'BETWEEN on large table (log) - range scan + kind filter', lambda s: f"""
            SELECT log_id, node_id, kind, val, occurred_at
            FROM log
            WHERE occurred_at BETWEEN '{s._param_date_minus_days(14, 30)}' AND '{s._param_date_minus_days(1,  13)}'
              AND kind = {s._param_log_kind()}
            ORDER BY occurred_at
            LIMIT 1000
        """)

        self._query('adv-7', 'Many-column output - per-group breakdown with multiple FILTER aggregates', lambda s: f"""
            SELECT n.grp_id,
                   COUNT(*)                                           AS node_count,
                   COUNT(*) FILTER (WHERE n.status = 0)              AS active_cnt,
                   COUNT(*) FILTER (WHERE n.status = 1)              AS inactive_cnt,
                   COUNT(*) FILTER (WHERE n.status = 2)              AS pending_cnt,
                   COUNT(*) FILTER (WHERE n.status = 3)              AS banned_cnt,
                   COUNT(*) FILTER (WHERE n.status = 4)              AS deleted_cnt,
                   AVG(n.val_int)                                     AS avg_val_int,
                   MIN(n.val_int)                                     AS min_val_int,
                   MAX(n.val_int)                                     AS max_val_int,
                   AVG(n.val_float)                                   AS avg_val_float,
                   COUNT(DISTINCT n.tag)                              AS distinct_tags
            FROM node n
            GROUP BY n.grp_id
            ORDER BY node_count DESC
        """)

        self._query('adv-8', 'ROLLUP - hierarchical aggregation (grp_id + status)', lambda s: f"""
            SELECT grp_id, status, COUNT(*) AS cnt, AVG(val_int) AS avg_val
            FROM node
            WHERE is_active = TRUE
            GROUP BY ROLLUP (grp_id, status)
            ORDER BY grp_id NULLS LAST, status NULLS LAST
        """)

        #endregion
        #region Deep multi-table joins (5+ tables / sources)

        self._query('graph-0', '5-way join: node + grp + parent-grp self-join + log + measure', lambda s: f"""
            SELECT n.node_id, n.tag, n.status,
                    g.name                     AS grp_name,
                    pg.name                    AS parent_grp,
                    COUNT(DISTINCT l.log_id)   AS log_count,
                    AVG(m.val)                 AS avg_measure
            FROM node n
            JOIN grp g        ON g.grp_id   = n.grp_id
            LEFT JOIN grp pg  ON pg.grp_id  = g.parent_id
            LEFT JOIN log l   ON l.node_id  = n.node_id
                                AND l.occurred_at >= '{s._param_date_minus_days(30, 90)}'
            LEFT JOIN measure m ON m.node_id = n.node_id
                                AND m.dim     = {s._param_dim()}
            WHERE g.grp_id = {s._param_grp_id()}
            GROUP BY n.node_id, n.tag, n.status, g.name, pg.name
            ORDER BY log_count DESC
            LIMIT 100
        """)

        self._query('graph-1', '6-way join: node + grp + parent-grp + doc + link + log', lambda s: f"""
            SELECT n.node_id, n.tag,
                    g.name                    AS grp,
                    pg.name                   AS parent_grp,
                    d.doc_id,
                    COUNT(DISTINCT lk.dst_id) AS out_degree,
                    COUNT(DISTINCT l.log_id)  AS log_count,
                    MAX(l.occurred_at)         AS last_event
            FROM node n
            JOIN grp g        ON g.grp_id  = n.grp_id
            LEFT JOIN grp pg  ON pg.grp_id = g.parent_id
            LEFT JOIN doc d   ON d.node_id = n.node_id
            LEFT JOIN link lk ON lk.src_id = n.node_id
            LEFT JOIN log l   ON l.node_id = n.node_id
                                AND l.kind    = {s._param_log_kind()}
            WHERE n.is_active = TRUE
                AND n.grp_id IN ({s._param_grp_ids(3, 8)})
            GROUP BY n.node_id, n.tag, g.name, pg.name, d.doc_id
            ORDER BY out_degree DESC, log_count DESC
            LIMIT 50
        """)

        def q_83(s: Self):
            """For-loop: 5 LEFT JOINs to measure, one per dim, pivoted into columns."""
            joins = '\n'.join(
                f"                LEFT JOIN measure m{d} ON m{d}.node_id = n.node_id AND m{d}.dim = {d}"
                for d in range(5)
            )
            agg_cols = ',\n                       '.join(
                f"AVG(m{d}.val) AS avg_dim{d}, COUNT(m{d}.measure_id) AS cnt_dim{d}"
                for d in range(5)
            )
            return f"""
                SELECT n.node_id, n.tag,
                       {agg_cols}
                FROM node n
{joins}
                WHERE n.grp_id    = {s._param_grp_id()}
                  AND n.is_active = TRUE
                GROUP BY n.node_id, n.tag
                ORDER BY n.node_id
                LIMIT 100
            """
        self._query('graph-2', 'Multi-dim pivot via 5 for-loop LEFT JOINs to measure (one per dim)', q_83)

        self._query('graph-3', '8-source join: node + grp + parent-grp + doc + link in/out degree + log + measure subqueries', lambda s: f"""
            SELECT n.node_id, n.tag, g.name AS grp, pg.name AS parent_grp,
                    d.created_at                    AS doc_ts,
                    COALESCE(lk_out.out_cnt, 0)    AS out_degree,
                    COALESCE(lk_in.in_cnt,   0)    AS in_degree,
                    COALESCE(la.log_cnt,      0)    AS log_cnt,
                    COALESCE(ma.measure_cnt,  0)    AS measure_cnt,
                    ma.avg_val
            FROM node n
            JOIN grp g        ON g.grp_id   = n.grp_id
            LEFT JOIN grp pg  ON pg.grp_id  = g.parent_id
            LEFT JOIN doc d   ON d.node_id  = n.node_id
            LEFT JOIN (
                SELECT src_id, COUNT(*) AS out_cnt FROM link GROUP BY src_id
            ) lk_out ON lk_out.src_id = n.node_id
            LEFT JOIN (
                SELECT dst_id, COUNT(*) AS in_cnt FROM link GROUP BY dst_id
            ) lk_in  ON lk_in.dst_id  = n.node_id
            LEFT JOIN (
                SELECT node_id, COUNT(*) AS log_cnt
                FROM   log
                WHERE  occurred_at >= '{s._param_date_minus_days(30, 90)}'
                GROUP BY node_id
            ) la ON la.node_id = n.node_id
            LEFT JOIN (
                SELECT node_id, COUNT(*) AS measure_cnt, AVG(val) AS avg_val
                FROM   measure
                WHERE  dim = {s._param_dim()}
                GROUP BY node_id
            ) ma ON ma.node_id = n.node_id
            WHERE n.grp_id IN ({s._param_grp_ids(3, 6)})
            ORDER BY out_degree DESC
            LIMIT 100
        """)

        #endregion
        #region Graph traversal via link

        def q_85(s: Self):
            """For-loop: builds an N-hop reachability chain by joining link to itself repeatedly."""
            n_hops = 3
            lk_joins = '\n'.join(
                f"                JOIN link lk{i + 1} ON lk{i + 1}.src_id = lk{i}.dst_id"
                for i in range(1, n_hops)
            )
            nid = s._param_node_id()
            return f"""
                SELECT DISTINCT lk{n_hops}.dst_id AS reachable_id,
                                dst.tag, dst.status, dst.grp_id
                FROM link lk1
{lk_joins}
                JOIN node dst ON dst.node_id = lk{n_hops}.dst_id
                WHERE lk1.src_id         = {nid}
                  AND lk{n_hops}.dst_id <> {nid}
                ORDER BY reachable_id
                LIMIT 500
            """
        self._query('graph-4', '3-hop reachability via for-loop link self-join chain', q_85)

        def q_86(s: Self):
            nid = s._param_node_id()
            return f"""
                SELECT nbr_id, tag, status, 'out' AS direction
                FROM (SELECT lk.dst_id AS nbr_id FROM link lk WHERE lk.src_id = {nid}) o
                JOIN node ON node_id = nbr_id
                UNION ALL
                SELECT nbr_id, tag, status, 'in' AS direction
                FROM (SELECT lk.src_id AS nbr_id FROM link lk WHERE lk.dst_id = {nid}) i
                JOIN node ON node_id = nbr_id
                ORDER BY nbr_id, direction
            """
        self._query('graph-5', 'Bidirectional 1-hop neighbors: UNION ALL of outgoing and incoming links', q_86)

        self._query('graph-6', 'Common out-neighbors of two nodes via INTERSECT', lambda s: f"""
            SELECT nbr.node_id, nbr.tag, nbr.grp_id
            FROM (
                SELECT dst_id AS node_id FROM link WHERE src_id = {s._param_int('node_id_1', 1, s._counts.node)}
                INTERSECT
                SELECT dst_id AS node_id FROM link WHERE src_id = {s._param_int('node_id_2', 1, s._counts.node)}
            ) common
            JOIN node nbr ON nbr.node_id = common.node_id
            ORDER BY nbr.node_id
        """)

        self._query('graph-7', 'Triangle detection: 3-hop cycle A->B->C->A starting from a given node', lambda s: f"""
            SELECT DISTINCT lk1.src_id AS a, lk2.src_id AS b, lk3.src_id AS c
            FROM link lk1
            JOIN link lk2 ON lk2.src_id = lk1.dst_id
            JOIN link lk3 ON lk3.src_id = lk2.dst_id
                            AND lk3.dst_id = lk1.src_id
            WHERE lk1.src_id = {s._param_node_id()}
            LIMIT 200
        """)

        self._query('graph-8', 'Hub analysis: in-degree vs out-degree with imbalance metric', lambda s: f"""
            SELECT n.node_id, n.tag,
                    COALESCE(out_d.cnt, 0) AS out_degree,
                    COALESCE(in_d.cnt,  0) AS in_degree,
                    COALESCE(out_d.cnt, 0) + COALESCE(in_d.cnt, 0) AS total_degree,
                    COALESCE(out_d.cnt, 0) - COALESCE(in_d.cnt, 0) AS degree_imbalance
            FROM node n
            LEFT JOIN (SELECT src_id, COUNT(*) AS cnt FROM link GROUP BY src_id) out_d
                    ON out_d.src_id = n.node_id
            LEFT JOIN (SELECT dst_id, COUNT(*) AS cnt FROM link GROUP BY dst_id) in_d
                    ON in_d.dst_id  = n.node_id
            WHERE COALESCE(out_d.cnt, 0) + COALESCE(in_d.cnt, 0) >= {s._param_int('min_degree', 5, 20)}
            ORDER BY total_degree DESC
            LIMIT 200
        """)

        #endregion
        #region Window function extremes

        self._query('window-adv-0', 'Named WINDOW clause shared by 7 window functions (avg, min, max, stddev, size, deviation, normalize)', lambda s: f"""
            SELECT node_id, grp_id, val_int,
                   AVG(val_int)    OVER w AS grp_avg,
                   MIN(val_int)    OVER w AS grp_min,
                   MAX(val_int)    OVER w AS grp_max,
                   STDDEV(val_int) OVER w AS grp_stddev,
                   COUNT(*)        OVER w AS grp_size,
                   val_int - AVG(val_int) OVER w AS deviation,
                   (val_int - MIN(val_int) OVER w)
                       / NULLIF(MAX(val_int) OVER w - MIN(val_int) OVER w, 0)
                       AS normalized
            FROM node
            WHERE is_active = TRUE
              AND grp_id IN ({s._param_grp_ids(5, 15)})
            WINDOW w AS (PARTITION BY grp_id)
            ORDER BY grp_id, val_int
            LIMIT 500
        """)

        self._query('window-adv-1', 'FIRST_VALUE / LAST_VALUE with explicit ROWS BETWEEN frames', lambda s: f"""
            SELECT log_id, node_id, occurred_at, val,
                   FIRST_VALUE(val) OVER (
                       PARTITION BY node_id ORDER BY occurred_at
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS first_val,
                   LAST_VALUE(val) OVER (
                       PARTITION BY node_id ORDER BY occurred_at
                       ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                   ) AS last_val,
                   val - FIRST_VALUE(val) OVER (
                       PARTITION BY node_id ORDER BY occurred_at
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS change_from_first
            FROM log
            WHERE node_id = {s._param_node_id()}
              AND val IS NOT NULL
            ORDER BY occurred_at
        """)

        self._query('window-adv-2', 'PERCENTILE_CONT and PERCENTILE_DISC at multiple quantiles per group', lambda s: f"""
            SELECT grp_id,
                   COUNT(*)                                               AS cnt,
                   AVG(val_int)                                           AS mean,
                   PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY val_int) AS p10,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY val_int) AS p25,
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY val_int) AS median,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY val_int) AS p75,
                   PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY val_int) AS p90,
                   PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY val_int) AS median_disc,
                   MAX(val_int) - MIN(val_int)                           AS range_val
            FROM node
            WHERE is_active = TRUE
            GROUP BY grp_id
            ORDER BY grp_id
        """)

        self._query('window-adv-3', 'Three different window frames in one query (cumulative, rolling-7, partition)', lambda s: f"""
            SELECT measure_id, node_id, recorded_at, val,
                   AVG(val) OVER (
                       PARTITION BY node_id ORDER BY recorded_at
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cumulative_avg,
                   AVG(val) OVER (
                       PARTITION BY node_id ORDER BY recorded_at
                       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                   ) AS rolling_7_avg,
                   AVG(val) OVER (PARTITION BY node_id) AS partition_avg,
                   val / NULLIF(AVG(val) OVER (PARTITION BY node_id), 0) AS ratio_to_avg
            FROM measure
            WHERE node_id = {s._param_node_id()}
              AND dim     = {s._param_dim()}
            ORDER BY recorded_at
        """)

        self._query('window-adv-4', 'Pareto filter: window-based running share used as outer-query predicate', lambda s: f"""
            SELECT node_id, tag, val_int, grp_id, running_share
            FROM (
                SELECT n.node_id, n.tag, n.val_int, n.grp_id,
                        SUM(n.val_int) OVER (
                            PARTITION BY n.grp_id ORDER BY n.val_int DESC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) * 1.0
                        / NULLIF(SUM(n.val_int) OVER (PARTITION BY n.grp_id), 0)
                        AS running_share
                FROM node n
                WHERE n.is_active = TRUE
            ) ranked
            WHERE running_share <= {s._param_float('pareto_share', 0.3, 0.7)}
            ORDER BY grp_id, val_int DESC
            LIMIT 500
        """)

        #endregion
        #region Complex CTE chains (4-5 levels)

        self._query('cte-adv-0', '4-CTE pipeline: raw agg -> node summary -> RANK per group -> top-K + final join', lambda s: f"""
            WITH
            raw_activity AS (
                SELECT node_id,
                        COUNT(*)             AS log_cnt,
                        SUM(val)             AS log_sum,
                        COUNT(DISTINCT kind) AS kind_diversity
                FROM   log
                WHERE  occurred_at >= '{s._param_date_minus_days(14, 60)}'
                    AND  val IS NOT NULL
                GROUP BY node_id
                HAVING COUNT(*) >= {s._param_int('min_log', 3, 10)}
            ),
            node_summary AS (
                SELECT n.node_id, n.tag, n.grp_id,
                        COALESCE(ra.log_cnt,        0) AS log_cnt,
                        COALESCE(ra.log_sum,        0) AS log_sum,
                        COALESCE(ra.kind_diversity, 0) AS kind_diversity
                FROM   node n
                JOIN   raw_activity ra ON ra.node_id = n.node_id
                WHERE  n.is_active = TRUE
            ),
            ranked_per_grp AS (
                SELECT *,
                        RANK() OVER (PARTITION BY grp_id ORDER BY log_cnt DESC) AS rnk
                FROM   node_summary
            ),
            top_nodes AS (
                SELECT * FROM ranked_per_grp WHERE rnk <= {s._param_int('top_k', 3, 10)}
            )
            SELECT tn.node_id, tn.tag, g.name AS grp,
                    tn.log_cnt, tn.log_sum, tn.kind_diversity, tn.rnk
            FROM   top_nodes tn
            JOIN   grp g ON g.grp_id = tn.grp_id
            ORDER BY tn.grp_id, tn.rnk
        """)

        self._query('cte-adv-1', 'CTE of active groups + LATERAL top-measure node per group', lambda s: f"""
            WITH active_grps AS (
                SELECT g.grp_id, g.name,
                        COUNT(n.node_id) AS node_count
                FROM   grp g
                JOIN   node n ON n.grp_id = g.grp_id
                WHERE  n.is_active = TRUE
                GROUP BY g.grp_id, g.name
                HAVING COUNT(n.node_id) >= {s._param_int('min_nodes', 10, 50)}
                ORDER BY node_count DESC
                LIMIT 20
            )
            SELECT ag.grp_id, ag.name, ag.node_count,
                    top_m.node_id     AS top_node,
                    top_m.total_val,
                    top_m.measure_cnt
            FROM active_grps ag
            CROSS JOIN LATERAL (
                SELECT n.node_id,
                        SUM(m.val)   AS total_val,
                        COUNT(*)     AS measure_cnt
                FROM   node n
                JOIN   measure m ON m.node_id = n.node_id
                WHERE  n.grp_id    = ag.grp_id
                    AND  n.is_active = TRUE
                    AND  m.dim       = {s._param_dim()}
                GROUP BY n.node_id
                ORDER BY total_val DESC
                LIMIT 1
            ) top_m
            ORDER BY ag.node_count DESC
        """)

        self._query('cte-adv-2', 'Recursive CTE with path string accumulation (||) + aggregated join', lambda s: f"""
            WITH RECURSIVE path_traversal AS (
                SELECT grp_id, name, parent_id,
                        name::TEXT AS path,
                        0          AS level
                FROM   grp
                WHERE  depth = 0
                UNION ALL
                SELECT g.grp_id, g.name, g.parent_id,
                        pt.path || ' > ' || g.name,
                        pt.level + 1
                FROM   grp g
                JOIN   path_traversal pt ON g.parent_id = pt.grp_id
            )
            SELECT p.grp_id, p.name, p.path, p.level,
                    COUNT(n.node_id)  AS node_count,
                    AVG(n.val_float)  AS avg_val_float
            FROM   path_traversal p
            LEFT JOIN node n ON n.grp_id = p.grp_id
            GROUP BY p.grp_id, p.name, p.path, p.level
            ORDER BY p.path
        """)

        def q_98(s: Self):
            gid = s._param_grp_id()
            return f"""
                WITH RECURSIVE
                ancestors AS (
                    SELECT grp_id, name, parent_id, 0 AS dist
                    FROM   grp WHERE grp_id = {gid}
                    UNION ALL
                    SELECT g.grp_id, g.name, g.parent_id, a.dist + 1
                    FROM   grp g JOIN ancestors a ON g.grp_id = a.parent_id
                ),
                descendants AS (
                    SELECT grp_id, name, parent_id, 0 AS dist
                    FROM   grp WHERE grp_id = {gid}
                    UNION ALL
                    SELECT g.grp_id, g.name, g.parent_id, d.dist + 1
                    FROM   grp g JOIN descendants d ON g.parent_id = d.grp_id
                )
                SELECT 'ancestor'   AS role, grp_id, name, dist FROM ancestors   WHERE dist > 0
                UNION ALL
                SELECT 'descendant' AS role, grp_id, name, dist FROM descendants WHERE dist > 0
                ORDER BY role, dist, grp_id
            """
        self._query('cte-adv-3', 'Two independent RECURSIVE CTEs in one block: ancestors + descendants of a group', q_98)

        #endregion
        #region Advanced subquery patterns

        def q_99(s: Self):
            dim = s._param_dim()
            return f"""
                SELECT n.node_id, n.tag, n.grp_id, n.val_int
                FROM node n
                WHERE n.val_int > (
                    SELECT AVG(n2.val_int)
                    FROM   node n2
                    WHERE  n2.grp_id = n.grp_id
                      AND  EXISTS (
                              SELECT 1 FROM measure m
                              WHERE  m.node_id = n2.node_id
                                AND  m.dim     = {dim}
                                AND  m.val     > (
                                    SELECT AVG(m2.val) FROM measure m2 WHERE m2.dim = {dim}
                                )
                          )
                )
                  AND n.is_active = TRUE
                ORDER BY n.val_int DESC
                LIMIT 100
            """
        self._query('nested-0', '3-level nested subquery: node > grp avg of nodes with above-avg measure', q_99)

        self._query('nested-1', 'ALL and ANY: val_int exceeds ALL group averages and ANY top-100 measure vals', lambda s: f"""
            SELECT node_id, tag, val_int
            FROM node
            WHERE val_int > ALL (
                SELECT AVG(val_int)
                FROM   node
                GROUP BY grp_id
                HAVING COUNT(*) >= {s._param_int('min_cnt', 20, 80)}
            )
              AND val_int > ANY (
                SELECT val
                FROM   measure
                WHERE  dim = {s._param_dim()}
                  AND  val IS NOT NULL
                ORDER BY val DESC
                LIMIT 100
            )
            ORDER BY val_int DESC
            LIMIT 100
        """)

        self._query('nested-2', 'Subqueries in HAVING: groups above global avg AND above average group size', lambda s: f"""
            SELECT grp_id, COUNT(*) AS cnt, AVG(val_int) AS avg_val
            FROM node
            WHERE is_active = TRUE
            GROUP BY grp_id
            HAVING AVG(val_int) > (
                SELECT AVG(val_int) FROM node WHERE is_active = TRUE
            )
                AND COUNT(*) > (
                SELECT AVG(sub_cnt)
                FROM (
                    SELECT COUNT(*) AS sub_cnt
                    FROM   node
                    WHERE  is_active = TRUE
                    GROUP BY grp_id
                ) t
            )
            ORDER BY avg_val DESC
            LIMIT 50
        """)

        self._query('nested-3', '5 correlated scalar subqueries in SELECT (per-node stats across log, measure, link)', lambda s: f"""
            SELECT n.node_id, n.tag, n.val_int,
                    (SELECT COUNT(*)  FROM log     WHERE node_id = n.node_id)                    AS total_logs,
                    (SELECT COUNT(*)  FROM log     WHERE node_id = n.node_id AND val IS NOT NULL) AS non_null_logs,
                    (SELECT MAX(val)  FROM measure WHERE node_id = n.node_id)                    AS max_measure,
                    (SELECT COUNT(*)  FROM link    WHERE src_id  = n.node_id)                    AS out_degree,
                    (SELECT COUNT(*)  FROM link    WHERE dst_id  = n.node_id)                    AS in_degree
            FROM node n
            WHERE n.grp_id = {s._param_grp_id()}
            ORDER BY total_logs DESC
            LIMIT 100
        """)

        #endregion
        #region For-loop generated queries

        def q_103(s: Self):
            """All 8 log kinds pivoted into COUNT + AVG columns via FILTER."""
            kind_cnt = '\n'.join(
                f"                       COUNT(*) FILTER (WHERE kind = {k})                     AS kind_{k}_cnt,"
                for k in range(8)
            )
            kind_avg = '\n'.join(
                f"                       AVG(val) FILTER (WHERE kind = {k} AND val IS NOT NULL) AS kind_{k}_avg,"
                for k in range(8)
            )
            return f"""
                SELECT node_id,
                       COUNT(*)          AS total_cnt,
{kind_cnt}
{kind_avg}
                       MIN(occurred_at)  AS first_event,
                       MAX(occurred_at)  AS last_event
                FROM log
                WHERE occurred_at >= '{s._param_date_minus_days(30, 180)}'
                GROUP BY node_id
                HAVING COUNT(*) >= {s._param_int('min_events', 3, 10)}
                ORDER BY total_cnt DESC
                LIMIT 200
            """
        self._query('pivot-0', 'Log pivot: all 8 kinds as FILTER aggregate columns (for-loop)', q_103)

        def q_104(s: Self):
            """UNION ALL of per-dim measure summaries - branches generated with for loop."""
            grp_ids = s._param_grp_ids(5, 20)
            branches = '\nUNION ALL\n'.join(
                f"""                SELECT {d} AS dim,
                       COUNT(*)    AS cnt,
                       AVG(val)    AS avg_val,
                       MIN(val)    AS min_val,
                       MAX(val)    AS max_val,
                       STDDEV(val) AS stddev_val
                FROM measure m
                JOIN node n ON n.node_id = m.node_id
                WHERE m.dim    = {d}
                  AND n.grp_id IN ({grp_ids})"""
                for d in range(5)
            )
            return f"""
                {branches}
                ORDER BY dim
            """
        self._query('pivot-1', 'UNION ALL of all 5 per-dim measure summaries (for-loop branches)', q_104)

        def q_105(s: Self):
            """Double CASE WHEN pivot: status label x val_int decile bucket cross-tabulation."""
            status_labels = ['active', 'inactive', 'pending', 'banned', 'deleted']
            status_case = '\n                            '.join(
                f"WHEN {k} THEN '{v}'" for k, v in enumerate(status_labels)
            )
            bucket_case = '\n                            '.join(
                f"WHEN val_int <= {100 * (i + 1)} THEN '{100 * i + 1:4d}-{100 * (i + 1):4d}'"
                for i in range(9)
            )
            return f"""
                SELECT status_label, val_bucket,
                       COUNT(*)       AS cnt,
                       AVG(val_float) AS avg_float,
                       AVG(val_int)   AS avg_int
                FROM (
                    SELECT
                        CASE status
                            {status_case}
                            ELSE 'unknown'
                        END AS status_label,
                        CASE
                            {bucket_case}
                            ELSE ' 901-1000'
                        END AS val_bucket,
                        val_float, val_int
                    FROM node
                    WHERE is_active = TRUE
                ) labelled
                GROUP BY status_label, val_bucket
                ORDER BY status_label, val_bucket
            """
        self._query('pivot-2', 'Double CASE pivot: status label x val_int bucket cross-tab (for-loop WHEN clauses)', q_105)

        self._query('set-op-4', 'INTERSECT chain: nodes satisfying 4 independent conditions across 3 tables', lambda s: f"""
            SELECT node_id FROM node    WHERE tag    = '{s._param_tag()}'
            INTERSECT
            SELECT node_id FROM node    WHERE grp_id = {s._param_grp_id()} AND is_active = TRUE
            INTERSECT
            SELECT node_id FROM log     WHERE kind   = {s._param_log_kind()} AND val IS NOT NULL
            INTERSECT
            SELECT node_id FROM measure WHERE dim    = {s._param_dim()}
            ORDER BY node_id
        """)

        #endregion
        #region GROUPING SETS, CUBE, extended ROLLUP

        self._query('rollup-0', 'CUBE(grp_id, status) with GROUPING() subtotal flags', lambda s: f"""
            SELECT grp_id, status,
                   GROUPING(grp_id)        AS is_grp_subtotal,
                   GROUPING(status)         AS is_status_subtotal,
                   GROUPING(grp_id, status) AS is_grand_total,
                   COUNT(*)                AS cnt,
                   AVG(val_int)            AS avg_val_int,
                   SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_cnt
            FROM node
            GROUP BY CUBE(grp_id, status)
            ORDER BY grp_id NULLS LAST, status NULLS LAST
        """)

        self._query('rollup-1', 'GROUPING SETS with explicit 4-level granularity + GROUPING() flags', lambda s: f"""
            SELECT m.dim, n.grp_id, m.recorded_at,
                    GROUPING(m.dim)                     AS is_dim_total,
                    GROUPING(n.grp_id)                  AS is_grp_total,
                    GROUPING(m.dim, n.grp_id, m.recorded_at) AS is_grand_total,
                    COUNT(*)                            AS cnt,
                    AVG(m.val)                          AS avg_val,
                    SUM(m.val)                          AS sum_val
            FROM measure m
            JOIN node n ON n.node_id = m.node_id
            WHERE n.is_active    = TRUE
                AND m.recorded_at >= '{s._param_date_minus_days(30, 180)}'
            GROUP BY GROUPING SETS (
                (m.dim, n.grp_id, m.recorded_at),
                (m.dim, n.grp_id),
                (m.dim),
                ()
            )
            ORDER BY m.dim NULLS LAST, n.grp_id NULLS LAST, m.recorded_at NULLS LAST
            LIMIT 2000
        """)

        self._query('rollup-2', '3-level ROLLUP: measure dim -> node grp_id -> node status', lambda s: f"""
            SELECT m.dim, n.grp_id, n.status,
                   COUNT(*)   AS cnt,
                   AVG(m.val) AS avg_val,
                   SUM(m.val) AS sum_val
            FROM measure m
            JOIN node n ON n.node_id = m.node_id
            WHERE m.recorded_at >= '{s._param_date_minus_days(30, 180)}'
            GROUP BY ROLLUP (m.dim, n.grp_id, n.status)
            ORDER BY m.dim NULLS LAST, n.grp_id NULLS LAST, n.status NULLS LAST
            LIMIT 2000
        """)

        #endregion
        #region PostgreSQL built-in functions & operators

        self._query('misc-0', 'TABLESAMPLE SYSTEM(10) for approximate fast scan', lambda s: f"""
            SELECT node_id, tag, val_int, val_float, status, grp_id
            FROM node TABLESAMPLE SYSTEM(10)
            WHERE val_int > {s._param_val_int()}
            ORDER BY val_int DESC
        """)

        self._query('misc-1', 'string_agg with ORDER BY: comma-separated tag list per group', lambda s: f"""
            SELECT grp_id,
                   COUNT(DISTINCT tag)                                  AS distinct_tags,
                   string_agg(DISTINCT tag, ', ' ORDER BY tag)          AS tag_list,
                   string_agg(node_id::TEXT, ',' ORDER BY val_int DESC) AS node_ids_by_val
            FROM node
            WHERE is_active = TRUE
              AND grp_id IN ({s._param_grp_ids(3, 8)})
            GROUP BY grp_id
            ORDER BY grp_id
        """)

        self._query('misc-2', 'array_agg + unnest roundtrip: collect then re-expand node IDs', lambda s: f"""
            SELECT node.grp_id, unnested_node_id, tag, val_int
            FROM (
                SELECT grp_id,
                       unnest(array_agg(node_id ORDER BY val_int DESC)) AS unnested_node_id
                FROM   node
                WHERE  is_active = TRUE
                  AND  grp_id    = {s._param_grp_id()}
                GROUP BY grp_id
            ) t
            JOIN node ON node_id = unnested_node_id
            ORDER BY node.grp_id, val_int DESC
            LIMIT 200
        """)

        self._query('misc-3', 'Regex match ~* (case-insensitive) on unindexed text column (note)', lambda s: f"""
            SELECT node_id, tag, note, status
            FROM node
            WHERE note ~* '^{chr(ord("a") + s._rng_int(5, 20))}'
                AND is_active = TRUE
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('misc-4', 'generate_series for weekly histogram of log events', lambda s: f"""
            SELECT gs.bucket,
                   COUNT(l.log_id) AS event_cnt,
                   AVG(l.val)      AS avg_val
            FROM generate_series(
                '{s._param_date_minus_days(90, 90)}'::TIMESTAMPTZ,
                now(),
                INTERVAL '7 days'
            ) AS gs(bucket)
            LEFT JOIN log l ON l.occurred_at >= gs.bucket
                           AND l.occurred_at <  gs.bucket + INTERVAL '7 days'
                           AND l.kind         = {s._param_log_kind()}
            GROUP BY gs.bucket
            ORDER BY gs.bucket
        """)

        self._query('misc-5', 'COALESCE chain: fill missing value from measure -> log avg -> val_float -> -1', lambda s: f"""
            SELECT n.node_id, n.tag,
                   COALESCE(
                       (SELECT AVG(m.val) FROM measure m
                        WHERE m.node_id = n.node_id AND m.dim = {s._param_dim()}),
                       (SELECT AVG(l.val) FROM log l
                        WHERE l.node_id = n.node_id AND l.val IS NOT NULL),
                       n.val_float,
                       -1.0
                   ) AS best_available_val
            FROM node n
            WHERE n.grp_id = {s._param_grp_id()}
            ORDER BY n.node_id
            LIMIT 100
        """)

        self._query('misc-6', 'GREATEST / LEAST across multiple columns + window-capped value', lambda s: f"""
            SELECT node_id, tag, val_int, val_float, status,
                   GREATEST(val_int, ROUND(val_float)::INT, status * 100)   AS greatest_val,
                   LEAST(val_int,    ROUND(val_float)::INT, status * 100 + 1) AS least_val,
                   GREATEST(val_int::FLOAT8, val_float)                     AS max_numeric,
                   ROUND(AVG(val_int) OVER (PARTITION BY grp_id))::INT      AS grp_avg_int,
                   LEAST(val_int,
                       ROUND(AVG(val_int) OVER (PARTITION BY grp_id))::INT + 50
                   ) AS val_capped_at_grp_avg_plus_50
            FROM node
            WHERE grp_id IN ({s._param_grp_ids(3, 8)})
            ORDER BY greatest_val DESC
            LIMIT 200
        """)

        self._query('misc-7', 'NULLIF to avoid division by zero in ratio calculations', lambda s: f"""
            SELECT g.grp_id, g.name,
                   COUNT(n.node_id)                                       AS total_nodes,
                   SUM(CASE WHEN n.is_active THEN 1 ELSE 0 END)           AS active_nodes,
                   SUM(CASE WHEN n.is_active THEN 1 ELSE 0 END) * 1.0
                       / NULLIF(COUNT(n.node_id), 0)                      AS active_ratio,
                   SUM(n.val_int)                                         AS val_sum,
                   SUM(n.val_int) * 1.0
                       / NULLIF(SUM(CASE WHEN n.is_active THEN 1 ELSE 0 END), 0)
                       AS val_per_active_node
            FROM grp g
            LEFT JOIN node n ON n.grp_id = g.grp_id
            GROUP BY g.grp_id, g.name
            ORDER BY active_ratio DESC NULLS LAST
        """)

        #endregion
        #region Kitchen sink / extreme complexity

        self._query('misc-8', 'VALUES inline lookup table joined with node + log aggregation', lambda s: f"""
            SELECT n.node_id, n.tag, cat.label, cat.tier,
                    COUNT(l.log_id) AS log_cnt
            FROM (VALUES
                (0, 'active',   'gold'),
                (1, 'inactive', 'silver'),
                (2, 'pending',  'bronze'),
                (3, 'banned',   'iron'),
                (4, 'deleted',  'none')
            ) AS cat(status_code, label, tier)
            JOIN node n ON n.status = cat.status_code
            LEFT JOIN log l ON l.node_id    = n.node_id
                            AND l.occurred_at >= '{s._param_date_minus_days(30, 90)}'
            WHERE n.grp_id = {s._param_grp_id()}
            GROUP BY n.node_id, n.tag, cat.label, cat.tier
            ORDER BY cat.tier, log_cnt DESC
            LIMIT 200
        """)

        self._query('misc-9', 'Monster: 5 CTEs + 3 window funcs + 5-way join + COALESCE + HAVING', lambda s: f"""
            WITH
            log_stats AS (
                SELECT node_id,
                        COUNT(*)             AS log_cnt,
                        AVG(val)             AS log_avg,
                        MAX(val)             AS log_max,
                        COUNT(DISTINCT kind) AS kind_diversity
                FROM   log
                WHERE  val IS NOT NULL
                    AND  occurred_at >= '{s._param_date_minus_days(30, 90)}'
                GROUP BY node_id
                HAVING COUNT(*) >= {s._param_int('min_log', 3, 10)}
            ),
            measure_stats AS (
                SELECT node_id,
                        COUNT(*)  AS measure_cnt,
                        AVG(val)  AS measure_avg,
                        MAX(val)  AS measure_max
                FROM   measure
                WHERE  dim          = {s._param_dim()}
                    AND  recorded_at >= '{s._param_date_minus_days(60, 180)}'
                GROUP BY node_id
            ),
            enriched AS (
                SELECT n.node_id, n.tag, n.grp_id, n.val_int,
                        COALESCE(ls.log_cnt,        0) AS log_cnt,
                        COALESCE(ls.log_avg,        0) AS log_avg,
                        COALESCE(ls.kind_diversity, 0) AS kind_diversity,
                        COALESCE(ms.measure_cnt,    0) AS measure_cnt,
                        COALESCE(ms.measure_avg,    0) AS measure_avg
                FROM   node n
                JOIN   log_stats     ls ON ls.node_id = n.node_id
                LEFT JOIN measure_stats ms ON ms.node_id = n.node_id
                WHERE  n.is_active = TRUE
            ),
            ranked AS (
                SELECT *,
                        RANK()   OVER (PARTITION BY grp_id ORDER BY log_cnt  DESC)     AS log_rank,
                        RANK()   OVER (PARTITION BY grp_id ORDER BY measure_avg DESC)   AS meas_rank,
                        NTILE(4) OVER (ORDER BY log_avg * COALESCE(measure_avg, 0) DESC) AS quartile
                FROM enriched
            ),
            top_per_grp AS (
                SELECT * FROM ranked WHERE log_rank <= {s._param_int('top_k', 5, 15)}
            )
            SELECT tp.node_id, tp.tag, g.name AS grp, pg.name AS parent_grp,
                    tp.log_cnt, tp.log_avg, tp.kind_diversity,
                    tp.measure_cnt, tp.measure_avg,
                    tp.log_rank, tp.meas_rank, tp.quartile,
                    d.created_at AS doc_ts
            FROM   top_per_grp tp
            JOIN   grp g     ON g.grp_id  = tp.grp_id
            LEFT JOIN grp pg ON pg.grp_id = g.parent_id
            LEFT JOIN doc d  ON d.node_id = tp.node_id
            ORDER BY tp.grp_id, tp.log_rank
        """)

        def q_120(s: Self):
            """Per-link-kind graph stats: 5 UNION ALL branches generated by for loop."""
            min_weight = s._param_float('min_weight', 0.1, 0.5)
            branches = '\nUNION ALL\n'.join(
                f"""                SELECT {k}                         AS link_kind,
                       COUNT(*)               AS cnt,
                       AVG(weight)            AS avg_weight,
                       MAX(weight)            AS max_weight,
                       COUNT(DISTINCT src_id) AS distinct_srcs,
                       COUNT(DISTINCT dst_id) AS distinct_dsts
                FROM link
                WHERE kind   = {k}
                  AND weight >= {min_weight}"""
                for k in range(5)
            )
            return f"""
                {branches}
                ORDER BY link_kind
            """
        self._query('complex-0', 'Per-link-kind graph statistics via 5 UNION ALL branches (for-loop)', q_120)

        self._query('complex-1', 'Temporal correlation: first/last event times from log and measure per node', lambda s: f"""
            SELECT n.node_id, n.tag,
                   MIN(l.occurred_at)  AS first_log,
                   MAX(l.occurred_at)  AS last_log,
                   MIN(m.recorded_at)  AS first_measure,
                   MAX(m.recorded_at)  AS last_measure,
                   MAX(l.occurred_at) - MIN(l.occurred_at) AS log_span,
                   COUNT(DISTINCT l.kind)                   AS log_kind_cnt,
                   COUNT(DISTINCT m.dim)                    AS measure_dim_cnt
            FROM node n
            JOIN log     l ON l.node_id = n.node_id
            JOIN measure m ON m.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
            GROUP BY n.node_id, n.tag
            HAVING COUNT(DISTINCT l.kind) >= {s._param_int('min_kinds', 2, 5)}
            ORDER BY log_span DESC NULLS LAST
            LIMIT 100
        """)

        self._query('complex-2', 'Complex multi-column ORDER BY with NULLS FIRST/LAST and LATERAL latest log', lambda s: f"""
            SELECT n.node_id, n.tag, n.val_int, n.val_float, n.status, n.grp_id,
                   l.occurred_at AS latest_log_ts, l.val AS latest_log_val
            FROM node n
            LEFT JOIN LATERAL (
                SELECT occurred_at, val FROM log
                WHERE  node_id = n.node_id
                ORDER BY occurred_at DESC NULLS LAST
                LIMIT 1
            ) l ON TRUE
            WHERE n.is_active = TRUE
            ORDER BY n.grp_id        ASC  NULLS LAST,
                     n.status        ASC  NULLS LAST,
                     n.val_int       DESC NULLS FIRST,
                     l.occurred_at   DESC NULLS LAST,
                     n.node_id       ASC
            LIMIT 500
        """)

        self._query('complex-3', 'CTE + RANK window + LATERAL top-2 logs per top-K node per group', lambda s: f"""
            WITH grp_nodes AS (
                SELECT g.grp_id, g.name AS grp_name,
                        n.node_id, n.tag, n.val_int
                FROM grp g
                JOIN node n ON n.grp_id = g.grp_id
                WHERE n.is_active = TRUE
                    AND g.depth     = {s._param_depth()}
            ),
            windowed AS (
                SELECT *,
                        RANK() OVER (PARTITION BY grp_id ORDER BY val_int DESC) AS rk
                FROM grp_nodes
            )
            SELECT w.grp_id, w.grp_name, w.node_id, w.tag, w.val_int, w.rk,
                    top_l.log_id, top_l.kind, top_l.val AS log_val, top_l.occurred_at
            FROM windowed w
            CROSS JOIN LATERAL (
                SELECT log_id, kind, val, occurred_at
                FROM   log
                WHERE  node_id = w.node_id
                    AND  val IS NOT NULL
                ORDER BY val DESC
                LIMIT 2
            ) top_l
            WHERE w.rk <= {s._param_int('top_k', 3, 5)}
            ORDER BY w.grp_id, w.rk, top_l.val DESC
            LIMIT 500
        """)

        self._query('complex-4', 'Cascading agg chain: node->dim stats -> group meta-stats -> z-score outlier + log join', lambda s: f"""
            WITH
            node_dim_stats AS (
                SELECT n.node_id, n.grp_id, m.dim,
                        AVG(m.val)    AS avg_val,
                        STDDEV(m.val) AS stddev_val,
                        COUNT(*)      AS n_points
                FROM node n
                JOIN measure m ON m.node_id = n.node_id
                GROUP BY n.node_id, n.grp_id, m.dim
            ),
            grp_dim_stats AS (
                SELECT grp_id, dim,
                        AVG(avg_val)    AS meta_avg,
                        AVG(stddev_val) AS meta_stddev,
                        SUM(n_points)   AS total_points,
                        COUNT(*)        AS node_cnt
                FROM node_dim_stats
                GROUP BY grp_id, dim
            ),
            outliers AS (
                SELECT nds.node_id, nds.grp_id, nds.dim,
                        nds.avg_val, gds.meta_avg, gds.meta_stddev,
                        (nds.avg_val - gds.meta_avg)
                            / NULLIF(gds.meta_stddev, 0) AS z_score
                FROM node_dim_stats nds
                JOIN grp_dim_stats  gds
                    ON gds.grp_id = nds.grp_id AND gds.dim = nds.dim
                WHERE ABS(
                    (nds.avg_val - gds.meta_avg) / NULLIF(gds.meta_stddev, 0)
                ) > {s._param_float('z_thresh', 1.5, 3.0)}
            )
            SELECT o.node_id, n.tag, g.name AS grp, o.dim,
                    o.avg_val, o.meta_avg, o.meta_stddev, o.z_score,
                    COUNT(l.log_id) AS recent_logs
            FROM outliers o
            JOIN node n ON n.node_id = o.node_id
            JOIN grp  g ON g.grp_id  = o.grp_id
            LEFT JOIN log l ON l.node_id    = o.node_id
                            AND l.occurred_at >= '{s._param_date_minus_days(30, 90)}'
            GROUP BY o.node_id, n.tag, g.name, o.dim,
                        o.avg_val, o.meta_avg, o.meta_stddev, o.z_score
            ORDER BY ABS(o.z_score) DESC
            LIMIT 100
        """)

        def q_125(s: Self):
            """Per-link-kind pivot + dst-node stats grouped by destination group."""
            kind_cols = '\n'.join(
                f"                       COUNT(*) FILTER (WHERE lk.kind = {k}) AS kind_{k}_links,"
                for k in range(5)
            )
            return f"""
                SELECT dst.grp_id,
                       g.name AS grp_name,
                       COUNT(DISTINCT lk.dst_id)     AS reachable_nodes,
{kind_cols}
                       AVG(lk.weight)                AS avg_weight,
                       MIN(lk.weight)                AS min_weight,
                       MAX(lk.weight)                AS max_weight,
                       AVG(dst.val_int)              AS avg_dst_val_int,
                       COUNT(DISTINCT dst.status)    AS dst_status_variety
                FROM link lk
                JOIN node dst ON dst.node_id = lk.dst_id
                JOIN grp  g   ON g.grp_id    = dst.grp_id
                WHERE lk.src_id = {s._param_node_id()}
                GROUP BY dst.grp_id, g.name
                ORDER BY reachable_nodes DESC
            """
        self._query('complex-5', 'Neighbor analysis: link-kind FILTER pivot + dst-node stats by dest group (for-loop)', q_125)

        self._query('complex-6', 'Compound set ops: (grp-nodes UNION src-links) EXCEPT (inactive UNION low-weight dsts)', lambda s: f"""
            (
                SELECT node_id FROM node WHERE grp_id = {s._param_grp_id()}
                UNION
                SELECT src_id AS node_id FROM link WHERE kind = {s._param_link_kind()}
            )
            EXCEPT
            (
                SELECT node_id FROM node WHERE is_active = FALSE
                UNION
                SELECT dst_id AS node_id FROM link WHERE weight < {s._param_float('low_weight', 0.1, 0.3)}
            )
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('complex-7', 'Node self-join: find pairs in same group with same status and similar val_int', lambda s: f"""
            SELECT a.node_id AS node_a, b.node_id AS node_b,
                    a.tag     AS tag_a,  b.tag     AS tag_b,
                    a.val_int AS val_a,  b.val_int AS val_b,
                    b.val_int - a.val_int              AS val_diff,
                    ABS(a.val_float - b.val_float)     AS float_dist
            FROM node a
            JOIN node b ON b.grp_id  = a.grp_id
                        AND b.node_id > a.node_id
                        AND b.status  = a.status
            WHERE a.grp_id    = {s._param_grp_id()}
                AND a.is_active = TRUE
                AND b.is_active = TRUE
                AND ABS(a.val_int - b.val_int) < {s._param_int('max_val_diff', 10, 50)}
            ORDER BY float_dist
            LIMIT 200
        """)

        self._query('complex-8', 'Linear regression trend via REGR_SLOPE / REGR_R2 on measure time series', lambda s: f"""
            SELECT node_id, dim,
                   COUNT(*)                                                 AS n_points,
                   REGR_SLOPE(val, EXTRACT(EPOCH FROM recorded_at))         AS slope,
                   REGR_INTERCEPT(val, EXTRACT(EPOCH FROM recorded_at))     AS intercept,
                   REGR_R2(val, EXTRACT(EPOCH FROM recorded_at))            AS r_squared,
                   MIN(recorded_at) AS first_date,
                   MAX(recorded_at) AS last_date
            FROM measure
            WHERE dim        IN ({s._param_int('dim_a', 0, 2)}, {s._param_int('dim_b', 2, 4)})
              AND recorded_at >= '{s._param_date_minus_days(90, 365)}'
            GROUP BY node_id, dim
            HAVING REGR_COUNT(val, EXTRACT(EPOCH FROM recorded_at)) >= {s._param_int('min_pts', 5, 20)}
            ORDER BY ABS(REGR_SLOPE(val, EXTRACT(EPOCH FROM recorded_at))) DESC NULLS LAST
            LIMIT 200
        """)

        self._query('complex-9', 'Latest state per node: two DISTINCT ON subqueries (log + measure) joined to node', lambda s: f"""
            SELECT n.node_id, n.tag, g.name AS grp,
                   ll.kind        AS last_log_kind,
                   ll.val         AS last_log_val,
                   ll.occurred_at AS last_log_ts,
                   lm.val         AS last_measure_val,
                   lm.recorded_at AS last_measure_date,
                   lm.dim         AS last_measure_dim
            FROM node n
            JOIN grp g ON g.grp_id = n.grp_id
            LEFT JOIN (
                SELECT DISTINCT ON (node_id)
                       node_id, kind, val, occurred_at
                FROM   log
                ORDER BY node_id, occurred_at DESC
            ) ll ON ll.node_id = n.node_id
            LEFT JOIN (
                SELECT DISTINCT ON (node_id)
                       node_id, val, recorded_at, dim
                FROM   measure
                ORDER BY node_id, recorded_at DESC
            ) lm ON lm.node_id = n.node_id
            WHERE n.grp_id    = {s._param_grp_id()}
              AND n.is_active = TRUE
            ORDER BY n.node_id
            LIMIT 200
        """)

        self._query('complex-10', 'Full graph degree analysis: in/out-degree CTEs + z-score + hub/leaf role classification', lambda s: f"""
            WITH
            out_degrees AS (
                SELECT src_id AS node_id, COUNT(*) AS out_deg FROM link GROUP BY src_id
            ),
            in_degrees AS (
                SELECT dst_id AS node_id, COUNT(*) AS in_deg  FROM link GROUP BY dst_id
            ),
            node_degree AS (
                SELECT n.node_id, n.tag, n.grp_id,
                        COALESCE(od.out_deg, 0) AS out_deg,
                        COALESCE(id_.in_deg,  0) AS in_deg,
                        COALESCE(od.out_deg, 0) + COALESCE(id_.in_deg, 0) AS total_deg
                FROM node n
                LEFT JOIN out_degrees od  ON od.node_id  = n.node_id
                LEFT JOIN in_degrees  id_ ON id_.node_id = n.node_id
            ),
            degree_stats AS (
                SELECT AVG(total_deg)    AS avg_deg,
                        STDDEV(total_deg) AS stddev_deg,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_deg) AS median_deg,
                        MAX(total_deg)    AS max_deg
                FROM node_degree
            )
            SELECT nd.node_id, nd.tag, nd.grp_id,
                    nd.out_deg, nd.in_deg, nd.total_deg,
                    (nd.total_deg - ds.avg_deg) / NULLIF(ds.stddev_deg, 0) AS degree_z,
                    CASE
                        WHEN nd.total_deg > ds.avg_deg + 2 * ds.stddev_deg THEN 'hub'
                        WHEN nd.total_deg < ds.avg_deg - ds.stddev_deg      THEN 'leaf'
                        ELSE 'normal'
                    END AS node_role
            FROM node_degree nd
            CROSS JOIN degree_stats ds
            ORDER BY nd.total_deg DESC
            LIMIT 200
        """)

        #endregion
        #region Bitmap scans

        # Plan-node targets for ML-based query-time estimation
        #
        # Each query below is designed to exercise a specific executor
        # plan node type or a structural property that the neural network
        # must learn to cost accurately.  Plan node tags are noted in the
        # query title so the dataset can be labelled during preprocessing.

        # BitmapOr: the planner combines two btree index scans when OR
        # spans two separately-indexed columns and a seq scan would be
        # more expensive.  Result depends on table stats; we also add a
        # LIMIT-free variant to maximise the chance PG picks bitmap.
        self._query('bitmap-0', 'BitmapOr: OR across node(tag) and node(grp_id) indexes', lambda s: f"""
            SELECT node_id, tag, grp_id, created_at
            FROM node
            WHERE tag = '{s._param_tag()}' OR grp_id = {s._param_grp_id()}
            ORDER BY node_id
        """)

        self._query('bitmap-1', 'BitmapAnd: created_at range AND grp_id equality - both indexed', lambda s: f"""
            SELECT node_id, tag, grp_id, created_at
            FROM node
            WHERE created_at BETWEEN '{s._param_date_minus_days(30, 90)}'
                             AND     '{s._param_date_minus_days(1,  29)}'
              AND grp_id = {s._param_grp_id()}
        """)

        self._query('bitmap-2', 'BitmapOr: three OR predicates each hitting node(tag) index', lambda s: f"""
            SELECT node_id, tag, grp_id
            FROM node
            WHERE tag = '{s._param_tag()}'
               OR tag = '{s._param_tag()}'
               OR tag = '{s._param_tag()}'
            ORDER BY node_id
        """)

        #endregion
        #region Index Only Scan

        # To get an Index Only Scan the projected columns must all live
        # inside the index and the visibility map must be current enough.
        # We project only the indexed columns to maximise the chance PG
        # uses the index without touching the heap.
        self._query('idx-only-0', 'Index Only Scan candidate: project only tag (node_tag btree index)', lambda s: f"""
            SELECT tag, COUNT(*) AS n
            FROM node
            WHERE tag BETWEEN 'T{s._rng_int(0, 300):04d}' AND 'T{s._rng_int(300, s._counts.n_tag - 1):04d}'
            GROUP BY tag
            ORDER BY tag
        """)

        self._query('idx-only-1', 'Index Only Scan candidate: DISTINCT grp_id via node_grp_id index', lambda s: f"""
            SELECT DISTINCT grp_id
            FROM node
            ORDER BY grp_id
        """)

        self._query('idx-only-2', 'Index Only Scan + count: grp_id COUNT via index without heap access', lambda s: f"""
            SELECT grp_id, COUNT(*) AS n
            FROM node
            WHERE grp_id BETWEEN {s._param_int('lo', 1, 100)} AND {s._param_int('hi', 100, 200)}
            GROUP BY grp_id
            ORDER BY grp_id
        """)

        #endregion
        #region Memoize (PG 14+)

        # Memoize appears on the inner side of a NL join when the planner
        # detects that the inner will be probed with a small number of
        # repeated outer keys.  We engineer this by using a tiny DISTINCT
        # outer driving that produces few unique keys.
        self._query('memoize-0', 'Memoize: small DISTINCT outer (few grp_id keys) NL-joined to grp inner', lambda s: f"""
            SELECT d.grp_id, g.name, g.priority, COUNT(d.node_id) AS cnt
            FROM (
                SELECT DISTINCT grp_id, node_id
                FROM   node
                WHERE  status = {s._param_status()}
                ORDER BY grp_id
                LIMIT 50
            ) d
            JOIN grp g ON g.grp_id = d.grp_id
            GROUP BY d.grp_id, g.name, g.priority
            ORDER BY d.grp_id
        """)

        self._query('memoize-1', 'Memoize: tiny VALUES outer NL-joined to node inner via PK', lambda s: f"""
            SELECT v.id, n.tag, n.grp_id, n.val_int
            FROM (VALUES {', '.join(f'({s._param_int(f"v{i}", 1, s._counts.node)})'
                          for i in range(10))}) AS v(id)
            JOIN node n ON n.node_id = v.id
            ORDER BY v.id
        """)

        #endregion
        #region Merge Join

        # A sort-merge join is favoured when both sides can be delivered
        # sorted on the join key cheaply (e.g. already ordered by an index
        # or when the query has a matching ORDER BY that the planner can
        # merge into).
        self._query('merge-join-0', 'Merge Join candidate: node PK order matches log FK order with outer ORDER BY', lambda s: f"""
            SELECT n.node_id, n.tag, l.log_id, l.kind, l.occurred_at
            FROM node n
            JOIN log l ON l.node_id = n.node_id
            WHERE n.grp_id = {s._param_grp_id()}
            ORDER BY n.node_id, l.occurred_at
            LIMIT 500
        """)

        self._query('merge-join-1', 'Merge Join candidate: grp self-join with pre-sorted inputs via depth index', lambda s: f"""
            SELECT g.grp_id, g.name, pg.name AS parent_name,
                   COUNT(n.node_id) AS node_cnt
            FROM grp g
            LEFT JOIN grp pg ON pg.grp_id = g.parent_id
            LEFT JOIN node n  ON n.grp_id  = g.grp_id
            GROUP BY g.grp_id, g.name, pg.name
            ORDER BY g.grp_id
        """)

        #endregion
        #region MATERIALIZED vs NOT MATERIALIZED CTEs

        # The MATERIALIZED keyword forces the CTE to be evaluated once
        # and stored; without it PG may inline the CTE as a sub-expression
        # and push predicates through it.  These two variants produce
        # different plan shapes even for identical logic.
        self._query('mat-cte-0', 'MATERIALIZED CTE: forces optimizer barrier - predicate cannot be pushed through', lambda s: f"""
            WITH log_summary AS MATERIALIZED (
                SELECT node_id, COUNT(*) AS log_cnt, AVG(val) AS avg_val
                FROM   log
                WHERE  val IS NOT NULL
                GROUP BY node_id
            )
            SELECT n.node_id, n.tag, ls.log_cnt, ls.avg_val
            FROM   node n
            JOIN   log_summary ls ON ls.node_id = n.node_id
            WHERE  n.is_active = TRUE
              AND  n.grp_id    = {s._param_grp_id()}
            ORDER BY ls.log_cnt DESC
            LIMIT 100
        """)

        self._query('mat-cte-1', 'NOT MATERIALIZED CTE: inlined - optimizer can push predicates and avoid full eval', lambda s: f"""
            WITH active_nodes AS NOT MATERIALIZED (
                SELECT node_id, tag, grp_id, val_int
                FROM   node
                WHERE  is_active = TRUE
            )
            SELECT an.node_id, an.tag, l.log_id, l.occurred_at
            FROM   active_nodes an
            JOIN   log l ON l.node_id = an.node_id
            WHERE  an.grp_id = {s._param_grp_id()}
            ORDER BY l.occurred_at DESC
            LIMIT 200
        """)

        #endregion
        #region Parallel plan nodes

        # Parallel plans (Gather / Gather Merge) appear when the table is
        # large enough and max_parallel_workers_per_gather > 0.  We write
        # full-table aggregates and large joins without LIMIT to maximise
        # the chance PG chooses a parallel plan.
        self._query('parallel-0', 'Parallel aggregate candidate: full-table multi-stat on 80k-row log table', lambda s: f"""
            SELECT
                COUNT(*)        AS n,
                SUM(val)        AS sum_val,
                AVG(val)        AS avg_val,
                STDDEV(val)     AS std_val,
                MIN(val)        AS min_val,
                MAX(val)        AS max_val,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY val) AS median_val
            FROM log
            WHERE val IS NOT NULL
        """)

        self._query('parallel-1', 'Parallel hash join + parallel agg: full node x log without any LIMIT', lambda s: f"""
            SELECT n.grp_id,
                   COUNT(l.log_id)  AS log_cnt,
                   AVG(l.val)       AS avg_val,
                   SUM(l.val)       AS sum_val
            FROM node n
            JOIN log l ON l.node_id = n.node_id
            WHERE l.val IS NOT NULL
            GROUP BY n.grp_id
            ORDER BY log_cnt DESC
        """)

        self._query('parallel-2', 'Parallel seq scan candidate: full-table scan of measure (150k rows) with math', lambda s: f"""
            SELECT dim,
                   COUNT(*)             AS n,
                   AVG(val)             AS avg_val,
                   STDDEV(val)          AS std_val,
                   CORR(val, EXTRACT(EPOCH FROM recorded_at::TIMESTAMPTZ)) AS time_corr
            FROM measure
            GROUP BY dim
            ORDER BY dim
        """)

        #endregion
        #region MergeAppend (sorted UNION ALL)

        # MergeAppend appears when UNION ALL combines sub-plans that
        # are each already sorted on the same key and an outer ORDER BY
        # on that key makes merging cheaper than a full sort.
        self._query('merge-append-0', 'MergeAppend candidate: UNION ALL of two pre-sorted time series with outer ORDER BY', lambda s: f"""
            (SELECT node_id, occurred_at::TIMESTAMPTZ AS ts, 'log'     AS src
                FROM   log     WHERE kind = {s._param_log_kind()} ORDER BY occurred_at)
            UNION ALL
            (SELECT node_id, recorded_at::TIMESTAMPTZ AS ts, 'measure' AS src
                FROM   measure WHERE dim  = {s._param_dim()} ORDER BY recorded_at)
            ORDER BY ts
            LIMIT 500
        """)

        #endregion
        #region LockRows node

        self._query('lock-0', 'LockRows: FOR UPDATE on single node row (exclusive lock per row)', lambda s: f"""
            SELECT node_id, tag, val_int, status
            FROM node
            WHERE node_id = {s._param_node_id()}
            FOR UPDATE
        """)

        self._query('lock-1', 'LockRows: FOR SHARE on grp row (shared lock, allows concurrent reads)', lambda s: f"""
            SELECT grp_id, name, priority
            FROM grp
            WHERE grp_id = {s._param_grp_id()}
            FOR SHARE
        """)

        self._query('lock-2', 'LockRows: FOR UPDATE SKIP LOCKED (work-queue dequeue pattern)', lambda s: f"""
            SELECT node_id, tag, status, grp_id
            FROM node
            WHERE status    = {s._param_status()}
              AND is_active = TRUE
            ORDER BY node_id
            LIMIT {s._param_limit(5, 3)}
            FOR UPDATE SKIP LOCKED
        """)

        #endregion
        #region JSONB operator nodes

        # These exercise the JSONB execution path and, if a GIN index
        # exists on the meta column, different index access methods.
        self._query('jsonb-op-0', 'JSONB @> containment operator (GIN index path if available)', lambda s: f"""
            SELECT doc_id, node_id, meta
            FROM doc
            WHERE meta @> '{{"lang": "en"}}'::JSONB
            ORDER BY doc_id
            LIMIT 100
        """)

        self._query('jsonb-op-1', "JSONB ? key-exists operator (per-key GIN lookup)", lambda s: f"""
            SELECT doc_id, node_id, meta->>'lang' AS lang
            FROM doc
            WHERE meta ? 'tags'
              AND (meta->>'version')::INT >= {s._param_int('min_ver', 1, 3)}
            ORDER BY doc_id
            LIMIT 100
        """)

        self._query('jsonb-op-2', 'JSONB #>> path extraction: non-index filter on nested scalar', lambda s: f"""
            SELECT doc_id,
                    meta #>> '{{tags,0}}'        AS first_tag,
                    (meta #>> '{{score}}')::FLOAT8 AS score
            FROM doc
            WHERE (meta #>> '{{score}}')::FLOAT8 > {s._param_float('score_thr', 2.0, 4.5)}
            ORDER BY score DESC
            LIMIT 100
        """)

        self._query('jsonb-op-3', 'jsonb_path_exists (JSONPath): filter using SQL/JSON path expression', lambda s: f"""
            SELECT doc_id, node_id, meta
            FROM doc
            WHERE jsonb_path_exists(meta, '$.score ? (@ > 3.5)')
            LIMIT 100
        """)

        #endregion
        #region Date/time functional filters and groupings

        # These force the planner to evaluate expressions at runtime rather
        # than using straight index scans, producing different cost curves.
        self._query('datetime-0', 'DATE_TRUNC in GROUP BY: expression grouping disables occurred_at index scan', lambda s: f"""
            SELECT DATE_TRUNC('day', occurred_at) AS day,
                    kind,
                    COUNT(*)   AS cnt,
                    AVG(val)   AS avg_val
            FROM log
            WHERE occurred_at >= '{s._param_date_minus_days(30, 90)}'
                AND val IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)

        self._query('datetime-1', 'EXTRACT(DOW) in WHERE: non-index-able functional filter forces seq scan', lambda s: f"""
            SELECT log_id, node_id, occurred_at, val
            FROM log
            WHERE EXTRACT(DOW FROM occurred_at) IN (1, 2, 3, 4, 5)
              AND val IS NOT NULL
            ORDER BY occurred_at DESC
            LIMIT 500
        """)

        self._query('datetime-2', 'age() comparison: functional expression disables index on occurred_at', lambda s: f"""
            SELECT log_id, node_id, occurred_at,
                   age(NOW(), occurred_at) AS event_age
            FROM log
            WHERE age(NOW(), occurred_at) < INTERVAL '{s._param_int("days", 7, 60)} days'
              AND val IS NOT NULL
            ORDER BY event_age ASC
            LIMIT 200
        """)

        self._query('datetime-3', 'Monthly measure aggregation via DATE_TRUNC: expression-based GROUP BY + sort', lambda s: f"""
            SELECT DATE_TRUNC('month', recorded_at) AS month,
                    dim,
                    COUNT(*)        AS n,
                    AVG(val)        AS avg_val,
                    SUM(val)        AS sum_val
            FROM measure
            WHERE recorded_at >= '{s._param_date_minus_days(180, 365)}'
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)

        #endregion
        #region Window function variants not yet covered

        self._query('win-frame-0', 'GROUPS BETWEEN frame (PG 11+): avg over N preceding peer groups', lambda s: f"""
            SELECT measure_id, node_id, dim, recorded_at, val,
                   AVG(val) OVER (
                       PARTITION BY node_id, dim
                       ORDER BY recorded_at
                       GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW
                   ) AS groups_2_avg
            FROM measure
            WHERE node_id = {s._param_node_id()}
            ORDER BY dim, recorded_at
        """)

        self._query('win-frame-1', 'RANGE BETWEEN interval frame (PG 11+): time-based rolling 7-day window', lambda s: f"""
            SELECT log_id, node_id, occurred_at, val,
                    AVG(val) OVER (
                        PARTITION BY node_id
                        ORDER BY occurred_at
                        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
                    ) AS rolling_7d_avg
            FROM log
            WHERE node_id IN ({s._param_node_ids(3, 8)})
                AND val IS NOT NULL
            ORDER BY node_id, occurred_at
        """)

        self._query('win-frame-2', 'CUME_DIST and PERCENT_RANK: two distribution functions in one WindowAgg', lambda s: f"""
            SELECT node_id, grp_id, val_int,
                   CUME_DIST()    OVER (PARTITION BY grp_id ORDER BY val_int) AS cume_dist,
                   PERCENT_RANK() OVER (PARTITION BY grp_id ORDER BY val_int) AS pct_rank
            FROM node
            WHERE is_active = TRUE
              AND grp_id IN ({s._param_grp_ids(3, 8)})
            ORDER BY grp_id, pct_rank DESC
            LIMIT 500
        """)

        self._query('win-frame-3', 'DENSE_RANK / RANK / ROW_NUMBER: three rank functions forcing single WindowAgg node', lambda s: f"""
            SELECT node_id, grp_id, val_int,
                   RANK()       OVER (ORDER BY val_int DESC)           AS rnk,
                   DENSE_RANK() OVER (ORDER BY val_int DESC)           AS drnk,
                   ROW_NUMBER() OVER (ORDER BY val_int DESC, node_id)  AS rn
            FROM node
            WHERE grp_id = {s._param_grp_id()}
            ORDER BY rnk
            LIMIT 200
        """)

        self._query('win-frame-4', 'NTH_VALUE at positions 1/3/5 with UNBOUNDED FOLLOWING frame', lambda s: f"""
            SELECT node_id, dim, recorded_at, val,
                   NTH_VALUE(val, 1) OVER w AS first_val,
                   NTH_VALUE(val, 3) OVER w AS third_val,
                   NTH_VALUE(val, 5) OVER w AS fifth_val
            FROM measure
            WHERE node_id = {s._param_node_id()}
            WINDOW w AS (
                PARTITION BY node_id, dim
                ORDER BY recorded_at
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            )
            ORDER BY dim, recorded_at
        """)

        self._query('win-frame-5', 'Conditional cumulative SUM via CASE inside window (FILTER not allowed in window)', lambda s: f"""
            SELECT log_id, node_id, occurred_at, kind, val,
                   SUM(CASE WHEN kind <= {s._param_int('kind_lo', 2, 5)}
                            THEN COALESCE(val, 0) END)
                       OVER (PARTITION BY node_id ORDER BY occurred_at) AS low_kind_cumsum,
                   SUM(CASE WHEN kind >  {s._param_int('kind_hi', 2, 6)}
                            THEN COALESCE(val, 0) END)
                       OVER (PARTITION BY node_id ORDER BY occurred_at) AS high_kind_cumsum
            FROM log
            WHERE node_id = {s._param_node_id()}
            ORDER BY occurred_at
        """)

        #endregion
        #region Sort / Distinct variants

        self._query('distinct-0', 'Sort Distinct: DISTINCT + ORDER BY - planner must sort then deduplicate', lambda s: f"""
            SELECT DISTINCT tag
            FROM node
            WHERE status = {s._param_status()}
            ORDER BY tag
        """)

        self._query('distinct-1', 'Hash Distinct candidate: DISTINCT without ORDER BY lets planner use hash', lambda s: f"""
            SELECT DISTINCT tag
            FROM node
            WHERE grp_id IN ({s._param_grp_ids(5, 20)})
        """)

        self._query('distinct-2', 'Multi-column DISTINCT: (status, grp_id) - higher cardinality for hash/sort choice', lambda s: f"""
            SELECT DISTINCT status, grp_id
            FROM node
            WHERE is_active = TRUE
            ORDER BY status, grp_id
        """)

        #endregion
        #region Incremental Sort (PG 13+)

        # Incremental Sort reuses a partial sort prefix from an index scan
        # and then sorts incrementally within each prefix group.
        self._query('incr-sort-0', 'Incremental Sort candidate: ORDER BY (grp_id, created_at, val_int) - only grp_id indexed', lambda s: f"""
            SELECT node_id, grp_id, created_at, val_int
            FROM node
            WHERE is_active = TRUE
            ORDER BY grp_id, created_at, val_int
            LIMIT 500
        """)

        self._query('incr-sort-1', 'Incremental Sort on measure: ORDER BY (node_id, dim, recorded_at) with partial index', lambda s: f"""
            SELECT measure_id, node_id, dim, recorded_at, val
            FROM measure
            WHERE val >= {s._param_float('val_lo', 0.0, 50.0)}
            ORDER BY node_id, dim, recorded_at
            LIMIT 1000
        """)

        #endregion
        #region Selectivity extremes

        # These probe the far ends of the selectivity spectrum so the NN
        # learns both the very-cheap and very-expensive regions of each
        # operator's cost curve.
        self._query('selectivity-0', 'Ultra-selective: three-column AND (status=4 + exact val_int + NOT active) -> ~0-3 rows', lambda s: f"""
            SELECT node_id, tag, val_int, val_float
            FROM node
            WHERE status    = 4
              AND val_int   = {s._param_val_int()}
              AND is_active = FALSE
        """)

        self._query('selectivity-1', 'Near-full scan: val_int > 1 AND < 999 returns ~98 % of node rows', lambda s: f"""
            SELECT COUNT(*) AS total,
                   AVG(val_float) AS avg_f,
                   STDDEV(val_int) AS std_i
            FROM node
            WHERE val_int > 1
              AND val_int < 999
        """)

        self._query('selectivity-2', 'Full-table projection with arithmetic: measures per-row computation cost (no filter)', lambda s: f"""
            SELECT node_id,
                   SQRT(ABS(val_float))                              AS sqrt_f,
                   val_int::FLOAT8 / 1000.0                          AS val_normalized,
                   tag || ':' || status::TEXT                        AS label,
                   LN(GREATEST(val_float, 1e-9))                    AS log_f,
                   CASE WHEN val_int % 3 = 0 THEN 'fizz'
                        WHEN val_int % 5 = 0 THEN 'buzz'
                        ELSE val_int::TEXT END                       AS fizzbuzz
            FROM node
            ORDER BY node_id
        """)

        #endregion
        #region Cardinality crossing points for join-type learning

        # The planner switches between NL / Hash / Merge join based on the
        # estimated row counts on each side.  These queries target the
        # boundary regions where the algorithm choice flips.
        self._query('join-size-0', 'Small x Large: 1 grp row x many node rows - likely Nested Loop', lambda s: f"""
            SELECT g.grp_id, g.name, g.priority,
                   n.node_id, n.tag, n.val_int
            FROM grp g
            JOIN node n ON n.grp_id = g.grp_id
            WHERE g.grp_id = {s._param_grp_id()}
            ORDER BY n.node_id
        """)

        self._query('join-size-1', 'Large x Large: full node x full log (parallel Hash Join candidate)', lambda s: f"""
            SELECT n.grp_id, n.status,
                   COUNT(l.log_id)     AS log_cnt,
                   SUM(l.val)          AS log_sum,
                   COUNT(m.measure_id) AS measure_cnt
            FROM node n
            LEFT JOIN log     l ON l.node_id = n.node_id
            LEFT JOIN measure m ON m.node_id = n.node_id
            GROUP BY n.grp_id, n.status
            ORDER BY log_cnt DESC
        """)

        self._query('join-size-2', 'Medium x Large: filtered node (~20 %) x full log - Hash Join expected', lambda s: f"""
            SELECT n.node_id, n.tag, COUNT(l.log_id) AS log_cnt
            FROM node n
            JOIN log l ON l.node_id = n.node_id
            WHERE n.status IN ({s._param_int('s1', 0, 1)}, {s._param_int('s2', 1, 3)})
              AND l.val IS NOT NULL
            GROUP BY n.node_id, n.tag
            ORDER BY log_cnt DESC
            LIMIT 200
        """)

        #endregion
        #region Miscellaneous plan nodes

        self._query('plan-0', 'Values Scan: literal VALUES table as CTE source (no heap access)', lambda s: f"""
            SELECT v.code, v.label, COUNT(n.node_id) AS cnt, AVG(n.val_int) AS avg_val
            FROM (VALUES
                (0, 'active'),
                (1, 'inactive'),
                (2, 'pending'),
                (3, 'banned'),
                (4, 'deleted')
            ) AS v(code, label)
            LEFT JOIN node n ON n.status = v.code
            GROUP BY v.code, v.label
            ORDER BY v.code
        """)

        self._query('plan-1', 'CASE in GROUP BY key: computed group-key forces seq scan + hash aggregate', lambda s: f"""
            SELECT CASE
                       WHEN val_int <  250 THEN 'q1'
                       WHEN val_int <  500 THEN 'q2'
                       WHEN val_int <  750 THEN 'q3'
                       ELSE                     'q4'
                   END AS quartile,
                   status,
                   COUNT(*)       AS cnt,
                   AVG(val_float) AS avg_float
            FROM node
            GROUP BY 1, 2
            ORDER BY quartile, status
        """)

        self._query('plan-2', 'Subquery Scan: derived table with expression column prevents predicate pushdown', lambda s: f"""
            SELECT sub.node_id, sub.doubled_val, sub.bucket, g.name AS grp_name
            FROM (
                SELECT node_id, grp_id,
                       val_int * 2 AS doubled_val,
                       CASE WHEN val_int > 500 THEN 'high' ELSE 'low' END AS bucket
                FROM node
                WHERE status = {s._param_status()}
            ) sub
            JOIN grp g ON g.grp_id = sub.grp_id
            WHERE sub.doubled_val > {s._param_int('dbl_thresh', 100, 1800)}
            ORDER BY sub.doubled_val DESC
            LIMIT 100
        """)

        self._query('plan-3', 'EXCEPT ALL (multiset difference): less common set op producing different SetOp node', lambda s: f"""
            SELECT node_id FROM log WHERE kind = {s._param_log_kind()}
            EXCEPT ALL
            SELECT node_id FROM log WHERE kind = {s._param_log_kind()} AND val IS NOT NULL
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('plan-4', 'INTERSECT ALL (multiset intersection): distinct plan from INTERSECT', lambda s: f"""
            SELECT node_id FROM node WHERE grp_id IN ({s._param_grp_ids(5, 10)})
            INTERSECT ALL
            SELECT node_id FROM log  WHERE kind = {s._param_log_kind()}
            ORDER BY node_id
            LIMIT 200
        """)

        self._query('plan-5', 'generate_series cross-joined to node: FunctionScan x SeqScan Cartesian product', lambda s: f"""
            SELECT gs.bucket,
                   COUNT(n.node_id) AS node_cnt,
                   AVG(n.val_int)   AS avg_val
            FROM generate_series(0, 9) AS gs(bucket)
            JOIN node n ON n.val_int / 100 = gs.bucket
            GROUP BY gs.bucket
            ORDER BY gs.bucket
        """)

        self._query('plan-6', 'Subquery in ORDER BY: correlated subquery forces per-row log count evaluation', lambda s: f"""
            SELECT n.node_id, n.tag, n.grp_id
            FROM node n
            WHERE n.is_active = TRUE
              AND n.grp_id    = {s._param_grp_id()}
            ORDER BY (SELECT COUNT(*) FROM log l WHERE l.node_id = n.node_id) DESC,
                     n.node_id
            LIMIT 50
        """)

        self._query('plan-7', 'EXISTS + NOT EXISTS together: two separate correlated semi-/anti-join plans', lambda s: f"""
            SELECT n.node_id, n.tag
            FROM node n
            WHERE EXISTS (
                SELECT 1 FROM log l
                WHERE l.node_id = n.node_id AND l.kind = {s._param_log_kind()}
            )
              AND NOT EXISTS (
                SELECT 1 FROM log l
                WHERE l.node_id = n.node_id AND l.kind = {s._param_log_kind()}
                  AND l.val IS NOT NULL AND l.val < {s._param_float('low_val', 0.0, 20.0)}
            )
            ORDER BY n.node_id
            LIMIT 200
        """)

        self._query('plan-8', 'Multi-level EXISTS nesting: outer EXISTS wraps an inner EXISTS', lambda s: f"""
            SELECT g.grp_id, g.name
            FROM grp g
            WHERE EXISTS (
                SELECT 1 FROM node n
                WHERE  n.grp_id = g.grp_id
                  AND  EXISTS (
                           SELECT 1 FROM log l
                           WHERE l.node_id = n.node_id
                             AND l.val     > {s._param_float('thresh', 50.0, 90.0)}
                       )
            )
            ORDER BY g.grp_id
        """)

        self._query('plan-9', 'LATERAL with filter on lateral result: lateral eval only for qualifying rows', lambda s: f"""
            SELECT n.node_id, n.tag, top_m.dim, top_m.val, top_m.recorded_at
            FROM node n
            JOIN LATERAL (
                SELECT dim, val, recorded_at
                FROM   measure
                WHERE  node_id = n.node_id
                ORDER BY val DESC
                LIMIT 3
            ) top_m ON top_m.val > {s._param_float('min_val', 50.0, 90.0)}
            WHERE n.grp_id    = {s._param_grp_id()}
              AND n.is_active = TRUE
            ORDER BY n.node_id, top_m.val DESC
            LIMIT 300
        """)

        #endregion

    def _register_write_queries(self):
        # Coverage map
        # ============
        # insert-0   Single-row VALUES INSERT + MAX subquery PK               Plan: ModifyTable -> Result (with InitPlan)
        # insert-1   Multi-row INSERT via generate_series                     Plan: ModifyTable -> ProjectSet -> FunctionScan
        # insert-2   INSERT ... SELECT with seq scan source (large batch)     Plan: ModifyTable -> Seq Scan (log)
        # insert-3   INSERT ... SELECT with index scan + filter               Plan: ModifyTable -> Index Scan (log node_id)
        # insert-4   INSERT ... SELECT with JOIN source (node x measure)      Plan: ModifyTable -> Hash/NL Join
        # insert-5   INSERT ... SELECT with aggregate source                  Plan: ModifyTable -> HashAggregate -> Seq Scan
        # insert-6   INSERT ... ON CONFLICT (PK) DO NOTHING                   Plan: ModifyTable with conflict check (no update path)
        # insert-7   INSERT ... ON CONFLICT DO UPDATE (upsert with condition) Plan: ModifyTable with conflict update subplan
        # insert-8   INSERT ... RETURNING                                     Plan: ModifyTable with Returning projection
        # insert-9   INSERT into link (composite PK) ON CONFLICT DO NOTHING   Plan: ModifyTable -> Result with composite conflict check
        # update-0   UPDATE by PK (single row)                                Plan: ModifyTable -> Index Scan (PK)
        # update-1   UPDATE by indexed column range                           Plan: ModifyTable -> Bitmap/Index Scan
        # update-2   UPDATE by non-indexed column (many rows)                 Plan: ModifyTable -> Seq Scan
        # update-3   UPDATE with correlated subquery in SET                   Plan: ModifyTable -> Seq Scan + SubPlan (per-row)
        # update-4   UPDATE with FROM clause (join-based update)              Plan: ModifyTable -> Hash Join (node driving log)
        # update-5   UPDATE with IN subquery in WHERE                         Plan: ModifyTable -> Hash Semi-Join
        # update-6   UPDATE with CTE source                                   Plan: ModifyTable -> CTE Scan
        # update-7   Large range UPDATE (many rows)                           Plan: ModifyTable -> Seq Scan (large modify)
        # update-8   UPDATE with EXISTS correlated subquery in WHERE          Plan: ModifyTable -> Seq Scan + SubPlan EXISTS
        # update-9   UPDATE ... RETURNING                                     Plan: ModifyTable with Returning projection
        # delete-0   DELETE by PK                                             Plan: ModifyTable -> Index Scan (PK)
        # delete-1   DELETE by indexed date range on large table              Plan: ModifyTable -> Bitmap Index Scan (occurred_at)
        # delete-2   DELETE by non-indexed column                             Plan: ModifyTable -> Seq Scan
        # delete-3   DELETE with IN subquery                                  Plan: ModifyTable -> Hash Semi-Join
        # delete-4   DELETE with USING clause (join-based delete)             Plan: ModifyTable -> Hash Join (USING)
        # delete-5   DELETE with EXISTS correlated subquery                   Plan: ModifyTable -> Seq Scan + SubPlan EXISTS
        # delete-6   DELETE with NOT EXISTS (anti-join)                       Plan: ModifyTable -> Hash Anti-Join
        # delete-7   DELETE with CTE selecting rows to remove                 Plan: ModifyTable -> CTE Scan
        # delete-8   DELETE ... RETURNING                                     Plan: ModifyTable with Returning projection
        # delete-9   Batched DELETE via IN (subquery with LIMIT)              Plan: ModifyTable -> Hash Join (inner = limited subquery)

        #region INSERT

        # single VALUES row; PK derived from MAX subquery
        # Produces an InitPlan sub-node for the MAX scalar subquery.
        self._write_query('insert-0', 'INSERT single log row via VALUES - PK from MAX subquery (InitPlan)', lambda s: f"""
            INSERT INTO log (log_id, node_id, kind, val, occurred_at)
            VALUES (
                (SELECT COALESCE(MAX(log_id), 0) + 1 FROM log),
                {s._param_node_id()},
                {s._param_log_kind()},
                {s._param_float('val', 0.0, 100.0)},
                NOW()
            )
        """)

        # N rows via generate_series - FunctionScan source
        self._write_query('insert-1', 'INSERT N rows via generate_series: ModifyTable -> ProjectSet -> FunctionScan', lambda s: f"""
            INSERT INTO log (log_id, node_id, kind, val, occurred_at)
            SELECT (SELECT COALESCE(MAX(log_id), 0) FROM log) + gs,
                   {s._param_node_id()},
                   {s._param_log_kind()},
                   gs * {s._param_float('multiplier', 0.5, 9.99)},
                   NOW() - (gs || ' minutes')::INTERVAL
            FROM generate_series(1, {s._param_int('n_rows', 5, 50)}) AS gs
        """)

        # INSERT ... SELECT with seq scan (large batch, no filter)
        self._write_query('insert-2', 'INSERT ... SELECT seq scan source (batch copy with rescaled val)', lambda s: f"""
            INSERT INTO log (log_id, node_id, kind, val, occurred_at)
            SELECT (SELECT COALESCE(MAX(log_id), 0) FROM log)
                        + ROW_NUMBER() OVER (ORDER BY log_id),
                    node_id,
                    {s._param_int('kind_dest', 0, 7)},
                    COALESCE(val, 0.0) * {s._param_float('scale', 0.8, 1.2)},
                    NOW()
            FROM log
            WHERE kind = {s._param_int('kind_src', 0, 7)}
                AND val  IS NOT NULL
            LIMIT {s._param_limit(10, 3)}
        """)

        # INSERT ... SELECT with index scan source (node_id filter)
        self._write_query('insert-3', 'INSERT ... SELECT index scan source (filtered by node_id index)', lambda s: f"""
            INSERT INTO log (log_id, node_id, kind, val, occurred_at)
            SELECT (SELECT COALESCE(MAX(log_id), 0) FROM log)
                        + ROW_NUMBER() OVER (ORDER BY log_id),
                    node_id,
                    {s._param_log_kind()},
                    COALESCE(val, 0.0) * 0.9,
                    NOW()
            FROM log
            WHERE node_id = {s._param_node_id()}
                AND val     IS NOT NULL
        """)

        # INSERT ... SELECT with JOIN source (node x measure)
        self._write_query('insert-4', 'INSERT ... SELECT with JOIN source: ModifyTable -> Hash/NL Join', lambda s: f"""
            INSERT INTO log (log_id, node_id, kind, val, occurred_at)
            SELECT (SELECT COALESCE(MAX(log_id), 0) FROM log)
                        + ROW_NUMBER() OVER (ORDER BY n.node_id),
                    n.node_id,
                    {s._param_log_kind()},
                    m.val * 0.5,
                    NOW()
            FROM node n
            JOIN measure m ON m.node_id = n.node_id
            WHERE n.grp_id    = {s._param_grp_id()}
                AND m.dim       = {s._param_dim()}
                AND n.is_active = TRUE
            LIMIT {s._param_limit(10, 3)}
        """)

        # INSERT ... SELECT with aggregate source
        self._write_query('insert-5', 'INSERT ... SELECT with aggregate source: ModifyTable -> HashAggregate -> Seq Scan', lambda s: f"""
            INSERT INTO measure (measure_id, node_id, dim, val, recorded_at)
            SELECT (SELECT COALESCE(MAX(measure_id), 0) FROM measure)
                        + ROW_NUMBER() OVER (ORDER BY node_id),
                    node_id,
                    {s._param_dim()},
                    AVG(val) * 1.05,
                    CURRENT_DATE
            FROM measure
            WHERE dim         = {s._param_dim()}
                AND recorded_at >= '{s._param_date_minus_days(30, 180)}'
            GROUP BY node_id
            HAVING COUNT(*) >= {s._param_int('min_cnt', 3, 10)}
        """)

        # ON CONFLICT DO NOTHING (target: doc primary key)
        # doc_id is loaded from data so conflict is almost certain,
        # exercising the conflict-check code path.
        self._write_query('insert-6', 'INSERT ... ON CONFLICT (doc PK) DO NOTHING: exercises conflict check path', lambda s: f"""
            INSERT INTO doc (doc_id, node_id, body, meta, created_at)
            VALUES (
                {s._param_doc_id()},
                {s._param_node_id()},
                'benchmark placeholder body',
                '{{"lang":"en","version":1,"tags":[],"score":3.0}}'::JSONB,
                NOW()
            )
            ON CONFLICT (doc_id) DO NOTHING
        """)

        # ON CONFLICT DO UPDATE (upsert with WHERE condition)
        self._write_query('insert-7', 'INSERT ... ON CONFLICT DO UPDATE (upsert): conflict update subplan with WHERE', lambda s: f"""
            INSERT INTO doc (doc_id, node_id, body, meta, created_at)
            VALUES (
                {s._param_doc_id()},
                {s._param_node_id()},
                'upserted body text for benchmark run',
                '{{"lang":"en","version":99,"score":5.0,"tags":["bench"]}}'::JSONB,
                NOW()
            )
            ON CONFLICT (doc_id) DO UPDATE
                SET meta = EXCLUDED.meta,
                    body = EXCLUDED.body
            WHERE doc.created_at < NOW() - INTERVAL '30 days'
        """)

        # INSERT ... RETURNING
        self._write_query('insert-8', 'INSERT single measure row ... RETURNING: ModifyTable with Returning projection', lambda s: f"""
            INSERT INTO measure (measure_id, node_id, dim, val, recorded_at)
            VALUES (
                (SELECT COALESCE(MAX(measure_id), 0) + 1 FROM measure),
                {s._param_node_id()},
                {s._param_dim()},
                {s._param_float('val', 0.0, 200.0)},
                CURRENT_DATE
            )
            RETURNING measure_id, node_id, dim, val, recorded_at
        """)

        # INSERT into link (composite PK) ON CONFLICT DO NOTHING
        self._write_query('insert-9', 'INSERT into link with composite PK ON CONFLICT DO NOTHING', lambda s: f"""
            INSERT INTO link (src_id, dst_id, kind, weight, created_at)
            VALUES (
                {s._param_int('src_id', 1, s._counts.node)},
                {s._param_int('dst_id', 1, s._counts.node)},
                {s._param_link_kind()},
                {s._param_weight()},
                NOW()
            )
            ON CONFLICT (src_id, dst_id) DO NOTHING
        """)

        #endregion
        #region UPDATE

        # single row by PK -> Index Scan (PK)
        self._write_query('update-0', 'UPDATE single node row by PK: ModifyTable -> Index Scan (PK lookup)', lambda s: f"""
            UPDATE node
            SET val_int   = LEAST(val_int + {s._param_int('delta', 1, 50)}, 1000),
                is_active = (status = 0)
            WHERE node_id = {s._param_node_id()}
        """)

        # range update by indexed column -> Bitmap/Index Scan
        self._write_query('update-1', 'UPDATE log rows by indexed node_id + kind: Bitmap/Index Scan modify', lambda s: f"""
            UPDATE log
            SET val = val * {s._param_float('scale', 0.9, 1.1)}
            WHERE node_id = {s._param_node_id()}
              AND kind    = {s._param_log_kind()}
              AND val IS NOT NULL
        """)

        # update by non-indexed column -> Seq Scan (many rows)
        self._write_query('update-2', 'UPDATE many node rows by non-indexed status column: Seq Scan modify', lambda s: f"""
            UPDATE node
            SET is_active = FALSE
            WHERE status    = {s._param_status()}
              AND is_active = TRUE
              AND val_int   < {s._param_val_int()}
        """)

        # correlated subquery in SET (SubPlan per row)
        self._write_query('update-3', 'UPDATE with correlated subquery in SET: per-row SubPlan (AVG measure)', lambda s: f"""
            UPDATE node
            SET val_float = COALESCE(
                (SELECT AVG(m.val)
                    FROM   measure m
                    WHERE  m.node_id = node.node_id
                    AND  m.dim     = {s._param_dim()}),
                val_float
            )
            WHERE node_id = {s._param_node_id()}
        """)

        # UPDATE with FROM clause (join-based update)
        self._write_query('update-4', 'UPDATE with FROM aggregate subquery (join-based update): ModifyTable -> Hash Join', lambda s: f"""
            UPDATE log
            SET val = log.val + bonus.avg_bonus
            FROM (
                SELECT node_id,
                        AVG(val) * 0.1 AS avg_bonus
                FROM   log
                WHERE  kind = {s._param_int('kind_src', 0, 6)}
                    AND  val  IS NOT NULL
                GROUP BY node_id
            ) bonus
            WHERE log.node_id = bonus.node_id
                AND log.kind    = {s._param_int('kind_dest', 0, 7)}
                AND log.val     IS NOT NULL
        """)

        # UPDATE with IN subquery -> Hash Semi-Join
        self._write_query('update-5', 'UPDATE with IN subquery on active-group nodes: ModifyTable -> Hash Semi-Join', lambda s: f"""
            UPDATE log
            SET kind = {s._param_int('new_kind', 0, 7)}
            WHERE node_id IN (
                SELECT node_id
                FROM   node
                WHERE  grp_id    = {s._param_grp_id()}
                  AND  is_active = TRUE
            )
              AND kind = {s._param_log_kind()}
        """)

        # UPDATE with CTE source
        self._write_query('update-6', 'UPDATE driven by CTE: ModifyTable -> CTE Scan (materialized target set)', lambda s: f"""
            WITH target AS (
                SELECT node_id
                FROM   node
                WHERE  grp_id = {s._param_grp_id()}
                    AND  status = {s._param_status()}
            )
            UPDATE node
            SET val_int = LEAST(val_int + {s._param_int('delta', 1, 20)}, 1000)
            WHERE node_id IN (SELECT node_id FROM target)
        """)

        # large range update on measure table (high-volume modify)
        self._write_query('update-7', 'UPDATE large range of measure rows: high-volume Seq Scan modify (many dirty pages)', lambda s: f"""
            UPDATE measure
            SET val = val * {s._param_float('scale', 0.95, 1.05)}
            WHERE recorded_at BETWEEN '{s._param_date_minus_days(60, 180)}' AND '{s._param_date_minus_days(1,   59)}'
              AND dim = {s._param_dim()}
        """)

        # UPDATE with EXISTS correlated subquery in WHERE
        self._write_query('update-8', 'UPDATE with EXISTS correlated subquery in WHERE: Seq Scan + SubPlan EXISTS', lambda s: f"""
            UPDATE node
            SET is_active = TRUE
            WHERE node_id = {s._param_node_id()}
              AND EXISTS (
                  SELECT 1
                  FROM   log l
                  WHERE  l.node_id = node.node_id
                    AND  l.kind    = {s._param_log_kind()}
                    AND  l.val     > {s._param_float('thresh', 50.0, 90.0)}
              )
        """)

        # UPDATE ... RETURNING
        self._write_query('update-9', 'UPDATE log vals ... RETURNING: ModifyTable with Returning output projection', lambda s: f"""
            UPDATE log
            SET val = val * {s._param_float('scale', 0.8, 1.2)}
            WHERE node_id = {s._param_node_id()}
              AND val     IS NOT NULL
            RETURNING log_id, node_id, kind, val, occurred_at
        """)

        #endregion
        #region DELETE

        # delete by PK -> Index Scan (PK)
        self._write_query('delete-0', 'DELETE single log row by PK: ModifyTable -> Index Scan (PK)', lambda s: f"""
            DELETE FROM log
            WHERE log_id = {s._param_int('log_id', 1, s._counts.log)}
        """)

        # delete by indexed date range on large table
        self._write_query('delete-1', 'DELETE log rows by indexed occurred_at range: Bitmap Index Scan on large table', lambda s: f"""
            DELETE FROM log
            WHERE occurred_at < '{s._param_date_minus_days(180, 365)}'
              AND kind         = {s._param_log_kind()}
        """)

        # delete by non-indexed column -> Seq Scan
        self._write_query('delete-2', 'DELETE measure rows by non-indexed val threshold: Seq Scan modify', lambda s: f"""
            DELETE FROM measure
            WHERE dim = {s._param_dim()}
              AND val < {s._param_float('low_val', 0.0, 10.0)}
        """)

        # delete with IN subquery -> Hash Semi-Join
        self._write_query('delete-3', 'DELETE with IN subquery (deleted-status nodes): ModifyTable -> Hash Semi-Join', lambda s: f"""
            DELETE FROM log
            WHERE node_id IN (
                SELECT node_id
                FROM   node
                WHERE  status = {s._param_status()}
            )
              AND val IS NULL
        """)

        # delete with USING clause (join-based delete)
        self._write_query('delete-4', 'DELETE with USING (join on node): ModifyTable -> Hash Join (USING path)', lambda s: f"""
            DELETE FROM log
            USING node n
            WHERE log.node_id = n.node_id
              AND n.status    = {s._param_status()}
              AND n.grp_id    = {s._param_grp_id()}
              AND log.val     IS NULL
        """)

        # delete with EXISTS correlated subquery
        self._write_query('delete-5', 'DELETE with EXISTS correlated subquery: Seq Scan + SubPlan EXISTS', lambda s: f"""
            DELETE FROM log
            WHERE node_id = {s._param_node_id()}
              AND EXISTS (
                  SELECT 1
                  FROM   node
                  WHERE  node_id   = log.node_id
                    AND  is_active = FALSE
              )
        """)

        # delete with NOT EXISTS -> Hash Anti-Join
        self._write_query('delete-6', 'DELETE with NOT EXISTS: rows with no active node reference (Hash Anti-Join)', lambda s: f"""
            DELETE FROM log
            WHERE NOT EXISTS (
                SELECT 1
                FROM   node
                WHERE  node_id   = log.node_id
                  AND  is_active = TRUE
            )
              AND occurred_at < '{s._param_date_minus_days(90, 365)}'
        """)

        # delete with CTE identifying rows to remove
        self._write_query('delete-7', 'DELETE driven by CTE (oldest N log rows for node): ModifyTable -> CTE Scan', lambda s: f"""
            WITH to_delete AS (
                SELECT log_id
                FROM   log
                WHERE  node_id = {s._param_node_id()}
                ORDER BY occurred_at ASC
                LIMIT  {s._param_limit(5, 3)}
            )
            DELETE FROM log
            WHERE log_id IN (SELECT log_id FROM to_delete)
        """)

        # delete ... RETURNING
        self._write_query('delete-8', 'DELETE single measure row ... RETURNING: ModifyTable with Returning projection', lambda s: f"""
            DELETE FROM measure
            WHERE measure_id = {s._param_int('measure_id', 1, s._counts.measure)}
            RETURNING measure_id, node_id, dim, val, recorded_at
        """)

        # batched DELETE via IN (subquery with LIMIT) - exercises
        #            the case where the deleted set is determined by a
        #            limited subquery (planner sees bounded inner scan).
        self._write_query('delete-9', 'Batched DELETE via IN (subquery with LIMIT): ModifyTable -> Hash Join (bounded inner)', lambda s: f"""
            DELETE FROM measure
            WHERE measure_id IN (
                SELECT measure_id
                FROM   measure
                WHERE  dim         = {s._param_dim()}
                    AND  recorded_at < '{s._param_date_minus_days(180, 730)}'
                ORDER BY measure_id
                LIMIT  {s._param_limit(50, 3)}
            )
        """)

        #endregion
