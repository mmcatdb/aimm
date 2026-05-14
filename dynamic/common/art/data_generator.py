import json
import math
from typing_extensions import override
from core.data_generator import DataGenerator, clamp_int, iso, print_counts

def export():
    return ArtDataGenerator()

class ArtDataGenerator(DataGenerator):
    # Tables
    # ======
    # grp     : small hierarchical groups (2-level tree)
    # node    : main entities; tag follows a Zipf distribution; status is skewed toward 0
    # link    : directed edges; src_id follows a power-law (hub nodes dominate)
    # log     : time-series events; node_id is uniform; ~20 % of val is NULL
    # measure : periodic metrics; node_id is top-heavy (hot nodes get more rows)
    # doc     : large documents; roughly one document per two nodes

    # Distributions
    # =============
    # - node.tag       Zipf (exponent 1.2), ~500 distinct values
    # - node.status    skewed [0.70, 0.15, 0.08, 0.04, 0.03]
    # - link.src_id    power-law (top 2 % are hubs with 10x higher weight)
    # - measure.node_id top 10 % of nodes have 5x weight

    def __init__(self):
        super().__init__('art')

    @override
    def _generate_data(self):
        c = self.ArtCounts(self)
        print('Counts:', print_counts(c))

        # Approximate file sizes for scales:            0       |  9       |
        self.__generate_grp(c.grp)                    # 8   kB  |  192 kB  |
        self.__generate_nodes(c.node, c.grp, c.n_tag) # 2   MB  |  1   GB  |
        self.__generate_links(c.link, c.node)         # 1.5 MB  |  1.6 GB  |
        self.__generate_logs(c.log, c.node)           # 4   MB  |  3.1 GB  |
        self.__generate_measures(c.measure, c.node)   # 4.4 MB  |  3.6 GB  |
        self.__generate_docs(c.doc, c.node)           # 13  MB  |  3.3 GB  |
        # JSON-based collections (MongoDB-specific). Won't be generated for large scales.
        if self.allow_large_kinds(self._scale):
            self.__generate_node_rich()               # 20  MB  |  12  GB  |
            self.__generate_grp_tree()                # 68  kB  |  2.3 MB  |
            self.__generate_event_log(c)              # 21  MB  |  15  GB  |
            self.__generate_buckets(c)                # 12  MB  |  3   GB  |

    class ArtCounts:
        def __init__(self, gen: ArtDataGenerator):
            self.grp       = gen._scaled(    200, 0.60)
            self.node      = gen._scaled( 10_000, 1.00)
            self.link      = gen._scaled( 30_000, 1.10)
            self.log       = gen._scaled( 80_000, 1.05)
            self.measure   = gen._scaled(150_000, 1.05)
            self.doc       = gen._scaled(  5_000, 0.90)
            self.n_tag     = gen._scaled(    500, 0.70)
            self.event_log = gen._scaled( 60_000, 1.05)
            self.bucket    = gen._scaled(  5_000, 0.90)

    @staticmethod
    def allow_large_kinds(scale) -> bool:
        """For larger scales, we can't reliably generate the larger JSON-based collections within memory and time constraints, so we skip them. All queries and loaders should skip them as well."""
        return scale <= 9.0

    def generate_counts(self, scale: float) -> ArtCounts:
        """Returns row counts for the given scale without generating any data."""
        self._reset(scale)
        return self.ArtCounts(self)

    def __generate_grp(self, n_grp: int) -> None:
        """2-level hierarchy: sqrt(n) roots (depth 0), rest are direct children (depth 1)."""
        f, w = self._open_csv_output('grp', ['grp_id', 'name', 'parent_id', 'depth', 'priority'])

        n_roots = clamp_int(int(round(math.sqrt(n_grp))), 5, 50)

        # Populate in-memory stores used by __generate_grp_tree()
        self._grp_store: dict[int, dict] = {}
        self._children_by_grp: dict[int, list[int]] = {}

        for grp_id in range(1, n_roots + 1):
            name = self._rng_word()
            priority = round(self._rng.random(), 4)
            w.writerow([
                grp_id,
                name,
                '',
                0,
                priority,
            ])
            self._grp_store[grp_id] = {'grp_id': grp_id, 'name': name, 'depth': 0, 'priority': priority, 'parent_id': None}
            self._children_by_grp[grp_id] = []

        for grp_id in range(n_roots + 1, n_grp + 1):
            parent_id = self._rng.randint(1, n_roots)
            name = self._rng_word()
            priority = round(self._rng.random(), 4)
            w.writerow([
                grp_id,
                name,
                parent_id,
                1,
                priority,
            ])
            self._grp_store[grp_id] = {'grp_id': grp_id, 'name': name, 'depth': 1, 'priority': priority, 'parent_id': parent_id}
            self._children_by_grp.setdefault(parent_id, []).append(grp_id)

        f.close()

    def __generate_nodes(self, n_node: int, n_grp: int, n_tag: int) -> None:
        """
        node.tag   : Zipf (most queries land on a few hot tags)
        node.status: [0=active 70%, 1=inactive 15%, 2=pending 8%, 3=banned 4%, 4=deleted 3%]
        node.grp_id: uniform across all groups
        node.val_int / val_float: uniform
        """
        f, w = self._open_csv_output('node', ['node_id', 'tag', 'val_int', 'val_float', 'note', 'status', 'grp_id', 'created_at', 'is_active'])

        tag_sampler = self._create_sampler([1.0 / ((i + 1) ** 1.2) for i in range(n_tag)])
        status_sampler = self._create_sampler([0.70, 0.15, 0.08, 0.04, 0.03])

        # Populate in-memory stores used by __generate_grp_tree() and __generate_node_rich()
        self._node_count_by_grp: dict[int, int] = {}
        self._active_count_by_grp: dict[int, int] = {}

        for node_id in range(1, n_node + 1):
            tag = f'T{tag_sampler.sample_index():04d}'
            val_int = self._rng.randint(1, 1000)
            val_float = round(self._rng.random(), 6)
            note = self._rng_text(10, 30)
            status = status_sampler.sample_index()
            grp_id = self._rng.randint(1, n_grp)
            created_at = self._rng_timestamp_since(3)
            is_active = status == 0
            w.writerow([
                node_id,
                tag,
                val_int,
                val_float,
                note,
                status,
                grp_id,
                iso(created_at),
                str(is_active).lower(),
            ])
            self._node_count_by_grp[grp_id] = self._node_count_by_grp.get(grp_id, 0) + 1
            if is_active:
                self._active_count_by_grp[grp_id] = self._active_count_by_grp.get(grp_id, 0) + 1

        f.close()

    def __generate_links(self, n_link: int, n_node: int) -> None:
        """
        src_id: power-law - top 2 % of nodes are hubs (10x base weight).
        dst_id: uniform.
        Composite PK (src_id, dst_id) enforces uniqueness via in-memory set.
        """
        f, w = self._open_csv_output('link', ['src_id', 'dst_id', 'kind', 'weight', 'created_at'])

        hot_n = max(1, n_node // 50)
        weights = []
        for i in range(n_node):
            if i < hot_n:
                weights.append(10.0 / max(1.0, (i + 1) ** 0.5))
            else:
                weights.append(1.0 / ((i + 1) ** 0.3))
        src_sampler = self._create_sampler(weights)

        used: set[tuple[int, int]] = set()
        written = 0
        attempts = 0
        max_attempts = n_link * 20

        while written < n_link and attempts < max_attempts:
            attempts += 1
            src_id = src_sampler.sample_index() + 1
            dst_id = self._rng.randint(1, n_node)
            if src_id == dst_id or (src_id, dst_id) in used:
                continue
            used.add((src_id, dst_id))
            w.writerow([
                src_id,
                dst_id,
                self._rng.randint(0, 4),
                round(self._rng.random(), 4),
                iso(self._rng_timestamp_since(2)),
            ])
            written += 1

        if written < n_link:
            print(f'  Warning: generated {written}/{n_link} links (collision budget exhausted).')

        f.close()

    def __generate_logs(self, n_log: int, n_node: int) -> None:
        """
        node_id: uniform across all nodes.
        val: ~80 % non-NULL, uniform [0, 100].
        occurred_at: uniform over last 2 years.
        """
        f, w = self._open_csv_output('log', ['log_id', 'node_id', 'kind', 'val', 'occurred_at'])

        for log_id in range(1, n_log + 1):
            node_id = self._rng.randint(1, n_node)
            kind = self._rng.randint(0, 7)
            val = round(self._rng.random() * 100, 4) if self._rng.random() > 0.20 else None
            occurred_at = self._rng_timestamp_since(2)
            w.writerow([
                log_id,
                node_id,
                kind,
                '' if val is None else val,
                iso(occurred_at),
            ])

        f.close()

    def __generate_measures(self, n_measure: int, n_node: int) -> None:
        """
        node_id: top-heavy - top 10 % of nodes carry 5x the weight.
        dim: uniform [0, 4], each dimension has a distinct value range:
          0 -> [0, 100]   1 -> [0, 100]   2 -> [0, 1 000]
          3 -> [0, 10 000]  4 -> [20, 100]
        recorded_at: slightly more recent dates via (rng^1.5) x 365 days ago.
        """
        f, w = self._open_csv_output('measure', ['measure_id', 'node_id', 'dim', 'val', 'recorded_at'])

        hot_n = max(1, n_node // 10)
        node_weights = [5.0 if i < hot_n else 1.0 for i in range(n_node)]
        node_sampler = self._create_sampler(node_weights)

        dim_ranges = [(0, 100), (0, 100), (0, 1000), (0, 10000), (20, 100)]

        for measure_id in range(1, n_measure + 1):
            node_id = node_sampler.sample_index() + 1
            dim = self._rng.randint(0, 4)
            lo, hi = dim_ranges[dim]
            val = round(lo + self._rng.random() * (hi - lo), 2)
            recorded_at = self._rng_timestamp_since(self._rng.random() ** 1.5).date()
            w.writerow([
                measure_id,
                node_id,
                dim,
                val,
                recorded_at.isoformat(),
            ])

        f.close()

    def __generate_docs(self, n_doc: int, n_node: int) -> None:
        """
        Assigns one doc to each of n_doc randomly chosen distinct nodes.
        body: large text (~200-500 words) to exercise wide-row queries.
        meta: JSONB with lang, version, tags, score.
        """
        f, w = self._open_csv_output('doc', ['doc_id', 'node_id', 'body', 'meta', 'created_at'])

        n_assign = min(n_doc, n_node)
        node_ids = self._rng.sample(range(1, n_node + 1), k=n_assign)

        for doc_id, node_id in enumerate(node_ids, 1):
            body = self._rng_text(200, 500)
            meta = json.dumps({
                'lang': self._rng_locale(),
                'version': self._rng.randint(1, 5),
                'tags': [self._rng_word() for _ in range(self._rng.randint(1, 5))],
                'score': round(self._rng.random() * 10, 2),
            }, ensure_ascii=False)
            created_at = self._rng_timestamp_since(2)
            w.writerow([
                doc_id,
                node_id,
                body,
                meta,
                iso(created_at),
            ])

        f.close()

    #region JSON collections

    LINK_LABELS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'critical', 'experimental', 'deprecated', 'important', 'temp']
    LINK_TIERS = ['bronze', 'silver', 'gold', 'platinum']

    def __generate_node_rich(self) -> None:
        """
        Deeply nested node documents (4-5 levels).  Built via two-pass CSV reading
        so we never hold 150 K measure rows in memory simultaneously.

        Structure:
          node_id, identity.{tag, status, is_active, labels[], flags.{is_hot, is_experimental, is_deprecated}},
          scores.{raw.{val_int, val_float}, composite, tier},
          context.{grp_id, grp.{...}, created_at, note},
          dimensions.d0..d4 (each null or {count, avg, min, max, readings[{val, ts}]}),
          connections.{outgoing[{dst_id, kind, weight, attrs.{label, strong, tags[]}}], out_degree, in_degree_est},
          audit.{schema_version, created_by, history[{ts, field, from_val, to_val, changed_by}]}
        """
        f, w = self._open_json_output('node_rich')

        # --- Pass 1: read measures ----------------------------------------
        measure_data: dict[int, dict[int, dict]] = {}  # node_id -> dim -> summary
        f_in, r = self._open_csv_input('measure', ['measure_id', 'node_id', 'dim', 'val', 'recorded_at'])

        for row in r:
            node_id, dim, val = int(row[1]), int(row[2]), float(row[3])
            ts = row[4]
            by_dim = measure_data.setdefault(node_id, {})
            e = by_dim.setdefault(dim, {'count': 0, 'sum': 0.0, 'min': 1e18, 'max': -1e18, 'readings': []})
            e['count'] += 1
            e['sum'] += val
            e['min'] = min(e['min'], val)
            e['max'] = max(e['max'], val)
            if len(e['readings']) < 5:
                e['readings'].append({'val': val, 'ts': ts})

        f_in.close()

        # --- Pass 2: read links -------------------------------------------
        link_data: dict[int, list[dict]] = {}   # src_id -> [out-links] capped at 8
        in_degree: dict[int, int] = {}
        f_in, r = self._open_csv_input('link', ['src_id', 'dst_id', 'kind', 'weight', 'created_at'])

        for row in r:
            src_id, dst_id, kind, weight = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            in_degree[dst_id] = in_degree.get(dst_id, 0) + 1
            lst = link_data.setdefault(src_id, [])
            if len(lst) < 8:
                lst.append({'dst_id': dst_id, 'kind': kind, 'weight': weight})

        f_in.close()

        # --- Pass 3: read nodes and produce rich documents ----------------
        attribute_labels = ['depends_on', 'related_to', 'blocks', 'follows', 'derived_from', 'references']

        f_in, r = self._open_csv_input('node', ['node_id', 'tag', 'val_int', 'val_float', 'note', 'status', 'grp_id', 'created_at', 'is_active'])

        for row in r:
            node_id    = int(row[0])
            tag        = row[1]
            val_int    = int(row[2])
            val_float  = float(row[3])
            note       = row[4]
            status     = int(row[5])
            grp_id     = int(row[6])
            created_at = row[7]
            is_active  = row[8].strip().lower() == 'true'

            grp = self._grp_store.get(grp_id, {'grp_id': grp_id, 'name': 'unknown', 'depth': 0, 'priority': 0.5, 'parent_id': None})

            composite  = round(val_int + val_float * 100, 2)
            tier_idx   = min(3, int(composite / 275))
            n_labels   = self._rng.randint(0, 3)
            labels     = self._rng.sample(self.LINK_LABELS, n_labels)

            # Build per-dimension summaries
            dim_data: dict[str, object] = {}
            node_m = measure_data.get(node_id, {})
            for d in range(5):
                e = node_m.get(d)
                if e and e['count'] > 0:
                    dim_data[f'd{d}'] = {
                        'count': e['count'],
                        'avg':   round(e['sum'] / e['count'], 4),
                        'min':   e['min'],
                        'max':   e['max'],
                        'readings': e['readings'],
                    }
                else:
                    dim_data[f'd{d}'] = None

            # Build embedded connections
            out_lks = link_data.get(node_id, [])
            connections = {
                'outgoing': [{
                    'dst_id': lk['dst_id'],
                    'kind':   lk['kind'],
                    'weight': lk['weight'],
                    'attrs':  {
                        'label':  self._rng.choice(attribute_labels),
                        'strong': lk['weight'] >= 0.7,
                        'tags':   [self._rng_word() for _ in range(self._rng.randint(0, 2))],
                    },
                } for lk in out_lks],
                'out_degree':    len(out_lks),
                'in_degree_est': in_degree.get(node_id, 0),
            }

            # Build audit history
            history = [{
                'ts':         self._rng_timestamp_since(2),
                'field':      self._rng.choice(['status', 'tag', 'val_int', 'grp_id']),
                'from_val':   self._rng.randint(0, 10),
                'to_val':     self._rng.randint(0, 10),
                'changed_by': f'u{self._rng.randint(1, 100):04d}',
            } for _ in range(self._rng.randint(0, 3))]

            w.writeobject({
                'node_id':  node_id,
                'identity': {
                    'tag':       tag,
                    'status':    status,
                    'is_active': is_active,
                    'labels':    labels,
                    'flags': {
                        'is_hot':          self._rng.random() < 0.10,
                        'is_experimental': self._rng.random() < 0.20,
                        'is_deprecated':   status == 4,
                    },
                },
                'scores': {
                    'raw': {'val_int': val_int, 'val_float': val_float},
                    'composite': composite,
                    'tier':      self.LINK_TIERS[tier_idx],
                },
                'context': {
                    'grp_id':     grp_id,
                    'grp':        grp,
                    'created_at': created_at,
                    'note':       note,
                },
                'dimensions':  dim_data,
                'connections': connections,
                'audit': {
                    'schema_version': 2,
                    'created_by':     self._rng.choice(['system', 'admin', 'import']),
                    'history':        history,
                },
            })

        f_in.close()
        f.close()

    def __generate_grp_tree(self) -> None:
        """
        One document per grp.  Includes materialized path, ancestor list, children list,
        and pre-computed node-count statistics (for fast subtree analytics).
        Depends on _grp_store, _children_by_grp, _node_count_by_grp populated earlier.
        """
        f, w = self._open_json_output('grp_tree')

        for grp_id, grp in self._grp_store.items():
            depth     = grp['depth']
            parent_id = grp['parent_id']

            if depth == 0:
                path      = f'/{grp_id}'
                ancestors = []
            else:
                path = f'/{parent_id}/{grp_id}'
                pg   = self._grp_store.get(parent_id)
                ancestors = [{
                    'grp_id':   parent_id,
                    'name':     pg['name']     if pg else '',
                    'depth':    0,
                    'priority': pg['priority'] if pg else 0.5,
                }]

            child_ids = self._children_by_grp.get(grp_id, [])
            children  = []
            for cid in child_ids[:20]:
                cg = self._grp_store.get(cid, {})
                children.append({
                    'grp_id':     cid,
                    'name':       cg.get('name', ''),
                    'priority':   cg.get('priority', 0.5),
                    'node_count': self._node_count_by_grp.get(cid, 0),
                })

            node_count   = self._node_count_by_grp.get(grp_id, 0)
            active_count = self._active_count_by_grp.get(grp_id, 0)

            w.writeobject({
                'grp_id':    grp_id,
                'name':      grp['name'],
                'depth':     depth,
                'priority':  grp['priority'],
                'path':      path,
                'parent_id': parent_id,
                'ancestors': ancestors,
                'children':  children,
                'stats': {
                    'node_count':        node_count,
                    'active_node_count': active_count,
                    'child_count':       len(child_ids),
                },
            })

        f.close()

    EVENT_TYPES = ['status_changed', 'link_added', 'measure_recorded', 'doc_updated', 'tag_changed', 'activated', 'deactivated']
    ROLES = ['admin', 'user', 'system', 'service']

    def __generate_event_log(self, c: 'ArtCounts') -> None:
        """
        Polymorphic event journal.  Each event_type has a distinct payload shape,
        enabling $type, $exists, and polymorphic-field aggregation patterns.
        """
        f, w = self._open_json_output('event_log')

        type_sampler = self._create_sampler([3.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0])

        for event_id in range(1, c.event_log + 1):
            node_id    = self._rng.randint(1, c.node)
            event_type = self.EVENT_TYPES[type_sampler.sample_index()]

            # Polymorphic payload
            if event_type == 'status_changed':
                from_s = self._rng.randint(0, 4)
                to_s   = self._rng.choice([x for x in range(5) if x != from_s])
                payload = {'from_status': from_s, 'to_status': to_s, 'reason': self._rng.choice(['manual', 'automated', 'timeout', 'admin_override'])}
            elif event_type == 'link_added':
                payload = {'src_id': node_id, 'dst_id': self._rng.randint(1, c.node), 'kind': self._rng.randint(0, 4), 'weight': round(self._rng.random(), 4)}
            elif event_type == 'measure_recorded':
                dim = self._rng.randint(0, 4)
                payload = {'dim': dim, 'val': round(self._rng.random() * 100, 2), 'prev_val': round(self._rng.random() * 100, 2) if self._rng.random() > 0.5 else None}
            elif event_type == 'doc_updated':
                changes = self._rng.sample(['body', 'meta.lang', 'meta.score', 'meta.version', 'meta.tags'], k=self._rng.randint(1, 3))
                payload = {'doc_id': self._rng.randint(1, max(1, c.doc)), 'changes': changes}
            elif event_type in ('activated', 'deactivated'):
                rule_id = self._rng.randint(1, 50) if self._rng.random() > 0.5 else None
                payload = {'trigger': self._rng.choice(['user', 'schedule', 'rule']), 'rule_id': rule_id}
            else:  # tag_changed
                payload = {'old_tag': f'T{self._rng.randint(0, 499):04d}', 'new_tag': f'T{self._rng.randint(0, 499):04d}'}

            w.writeobject({
                'event_id':   event_id,
                'node_id':    node_id,
                'event_type': event_type,
                'occurred_at': self._rng_timestamp_since(2),
                'actor': {
                    'user_id': f'u{self._rng.randint(1, 100):04d}',
                    'role':    self._rng.choice(self.ROLES),
                    'session': f'sess_{self._rng_word()}',
                },
                'payload': payload,
                'context': {
                    'ip_hash':    f'ip_{self._rng_word()}',
                    'request_id': f'req_{self._rng_word()}_{event_id}',
                    'trace_id':   f'trace_{self._rng.randint(1, 10000):06d}',
                },
            })

        f.close()

    def __generate_buckets(self, c: 'ArtCounts') -> None:
        """
        Bucket-pattern time-series: one document per (node, dim, hour).
        Stores pre-computed aggregate stats plus raw data[] array for $unwind-based queries.
        Enables $densify (gap-fill), re-bucketing, and quality-filtered analytics.
        """
        f, w = self._open_json_output('bucket')

        quality_levels = ['good', 'fair', 'poor', 'suspect']
        quality_sampler = self._create_sampler([0.70, 0.15, 0.10, 0.05])

        hot_n = max(1, c.node // 10)
        node_weights = [5.0 if i < hot_n else 1.0 for i in range(c.node)]
        node_sampler = self._create_sampler(node_weights)

        dim_ranges = [(0.0, 100.0), (0.0, 100.0), (0.0, 1000.0), (0.0, 10000.0), (20.0, 100.0)]

        for bucket_id in range(1, c.bucket + 1):
            node_id  = node_sampler.sample_index() + 1
            dim      = self._rng.randint(0, 4)
            lo, hi   = dim_ranges[dim]

            base_ts    = self._rng_timestamp_since(2)
            hour_start = base_ts.replace(minute=0, second=0, microsecond=0)

            n_data = self._rng.randint(20, 60)
            vals   = [round(lo + self._rng.random() * (hi - lo), 2) for _ in range(n_data)]

            min_v   = min(vals)
            max_v   = max(vals)
            sum_v   = round(sum(vals), 2)
            avg_v   = round(sum_v / n_data, 2)
            var_v   = sum((v - avg_v) ** 2 for v in vals) / n_data
            stddev_v = round(var_v ** 0.5, 2)

            interval = max(1, 3600 // n_data)
            data = [{
                'offset_s': i * interval + self._rng.randint(0, max(1, interval - 1)),
                'val':      vals[i],
                'quality':  quality_levels[quality_sampler.sample_index()],
            } for i in range(n_data)]

            w.writeobject({
                'bucket_id':  bucket_id,
                'node_id':    node_id,
                'dim':        dim,
                'hour_start': hour_start,
                'span_hours': 1,
                'count':      n_data,
                'min':        min_v,
                'max':        max_v,
                'sum':        sum_v,
                'avg':        avg_v,
                'stddev':     stddev_v,
                'data':       data,
            })

        f.close()

    #endregion
