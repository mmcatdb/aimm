from core.drivers import DriverType
from core.query import MongoQuery, MongoFindQuery, MongoAggregateQuery, MongoUpdateQuery, MongoDeleteQuery, MongoInsertQuery
from ...common.art.query_registry import ArtQueryRegistry

def export():
    return MongoArtQueryRegistry()

class MongoArtQueryRegistry(ArtQueryRegistry[MongoQuery]):

    def __init__(self):
        super().__init__(DriverType.MONGO)

    def _register_queries(self):
        self._register_find_node_queries()
        self._register_find_grp_queries()
        self._register_find_log_queries()
        self._register_find_measure_queries()
        self._register_find_link_queries()
        self._register_find_doc_queries()
        self._register_node_array_queries()
        self._register_node_nested_filter_queries()
        self._register_aggregation_queries()
        self._register_lookup_queries()
        self._register_graphlookup_queries()
        self._register_window_queries()
        self._register_facet_queries()
        self._register_event_log_queries()
        self._register_node_rich_queries()
        self._register_bucket_queries()
        self._register_grp_tree_queries()
        self._register_advanced_pattern_queries()
        self._register_write_queries()

    #region Find node

    def _register_find_node_queries(self):
        # PK / point lookups
        self._query('node-pk-0', 'node by id', lambda s: MongoFindQuery('node',
            filter={'node_id': s._param_node_id()}
        ))

        self._query('node-pk-1', 'node id + projection', lambda s: MongoFindQuery('node',
            filter={'node_id': s._param_node_id()},
            projection={'node_id': 1, 'tag': 1, 'status': 1, 'grp_id': 1, '_id': 0}
        ))

        # tag scans
        self._query('node-tag-0', 'nodes with tag', lambda s: MongoFindQuery('node',
            filter={'tag': s._param_tag()}
        ))

        self._query('node-tag-1', 'nodes with tag + status + sort', lambda s: MongoFindQuery('node',
            filter={'tag': s._param_tag(), 'status': s._param_status()},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('node-tag-2', 'nodes with tag IN list', lambda s: MongoFindQuery('node',
            filter={'tag': {'$in': [f'T{s._rng_int(0, s._counts.n_tag - 1):04d}' for _ in range(5)]}}
        ))

        self._query('node-tag-3', 'active nodes with tag + limit/skip', lambda s: MongoFindQuery('node',
            filter={'tag': s._param_tag(), 'is_active': True},
            sort={'node_id': 1},
            limit=s._param_limit(),
            skip=s._param_skip(0, 500)
        ))

        # status scans
        self._query('node-status-0', 'nodes by status', lambda s: MongoFindQuery('node',
            filter={'status': s._param_status()}
        ))

        self._query('node-status-1', 'nodes by status + sort + limit', lambda s: MongoFindQuery('node',
            filter={'status': s._param_status()},
            sort={'val_int': -1},
            limit=s._param_limit()
        ))

        self._query('node-status-2', 'nodes status IN list', lambda s: MongoFindQuery('node',
            filter={'status': {'$in': [0, 1]}}
        ))

        self._query('node-status-3', 'nodes status NOT IN list', lambda s: MongoFindQuery('node',
            filter={'status': {'$nin': [3, 4]}}
        ))

        self._query('node-status-4', 'nodes status range', lambda s: MongoFindQuery('node',
            filter={'status': {'$gte': 1, '$lte': 3}}
        ))

        # is_active
        self._query('node-active-0', 'active nodes', lambda s: MongoFindQuery('node',
            filter={'is_active': True}, sort={'created_at': -1}, limit=s._param_limit()
        ))

        self._query('node-active-1', 'inactive nodes by grp', lambda s: MongoFindQuery('node',
            filter={'is_active': False, 'grp_id': s._param_grp_id()},
            sort={'node_id': 1}
        ))

        # val_int scans
        self._query('node-val-int-0', 'nodes val_int equals', lambda s: MongoFindQuery('node',
            filter={'val_int': s._param_val_int()}
        ))

        self._query('node-val-int-1', 'nodes val_int range', lambda s: MongoFindQuery('node',
            filter={'val_int': {'$gte': 100, '$lte': 500}},
            sort={'val_int': 1},
            limit=s._param_limit()
        ))

        self._query('node-val-int-2', 'nodes val_int gt + status', lambda s: MongoFindQuery('node',
            filter={'val_int': {'$gt': s._param_val_int()}, 'status': 0},
            projection={'node_id': 1, 'val_int': 1, 'tag': 1, '_id': 0}
        ))

        # val_float scans
        self._query('node-val-float-0', 'nodes val_float range', lambda s: MongoFindQuery('node',
            filter={'val_float': {'$gte': 0.2, '$lte': 0.8}}
        ))

        self._query('node-val-float-1', 'nodes val_float lt', lambda s: MongoFindQuery('node',
            filter={'val_float': {'$lt': s._param_weight()}},
            sort={'val_float': -1},
            limit=s._param_limit()
        ))

        # grp_id scans
        self._query('node-grp-0', 'nodes by grp', lambda s: MongoFindQuery('node',
            filter={'grp_id': s._param_grp_id()}
        ))

        self._query('node-grp-1', 'nodes by grp + status', lambda s: MongoFindQuery('node',
            filter={'grp_id': s._param_grp_id(), 'status': s._param_status()},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('node-grp-2', 'nodes by grp IN list', lambda s: MongoFindQuery('node',
            filter={'grp_id': {'$in': s._param_grp_ids(3, 8)}}
        ))

        # date scans
        self._query('node-date-0', 'nodes created after date', lambda s: MongoFindQuery('node',
            filter={'created_at': {'$gte': s._param_date_minus_days(180, 365)}},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('node-date-1', 'nodes created in date range', lambda s: MongoFindQuery('node',
            filter={'created_at': {'$gte': s._param_date_minus_days(365, 730), '$lte': s._param_date_minus_days(0, 180)}},
            sort={'created_at': 1}
        ))

        self._query('node-date-2', 'active nodes created after date', lambda s: MongoFindQuery('node',
            filter={'is_active': True, 'created_at': {'$gte': s._param_date_minus_days(90, 270)}},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        # node_id IN list
        self._query('node-ids-0', 'nodes by id list (small)', lambda s: MongoFindQuery('node',
            filter={'node_id': {'$in': s._param_node_ids(3, 10)}}
        ))

        self._query('node-ids-1', 'nodes by id list (large)', lambda s: MongoFindQuery('node',
            filter={'node_id': {'$in': s._param_node_ids(20, 50)}}
        ))

        # compound filters
        self._query('node-compound-0', 'node tag+grp+status', lambda s: MongoFindQuery('node',
            filter={'tag': s._param_tag(), 'grp_id': s._param_grp_id(), 'status': {'$lte': 1}},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('node-compound-1', 'node val_int + is_active + sort val_float', lambda s: MongoFindQuery('node',
            filter={'val_int': {'$gte': s._param_val_int()}, 'is_active': True},
            sort={'val_float': -1},
            limit=s._param_limit()
        ))

        self._query('node-compound-2', 'node grp + val_float + date', lambda s: MongoFindQuery('node',
            filter={
                'grp_id': {'$in': s._param_grp_ids(2, 5)},
                'val_float': {'$gte': 0.5},
                'created_at': {'$gte': s._param_date_minus_days(180, 365)},
            },
            projection={'node_id': 1, 'tag': 1, 'val_float': 1, 'created_at': 1, '_id': 0}
        ))

        self._query('node-compound-3', 'node $or status/tag', lambda s: MongoFindQuery('node',
            filter={'$or': [{'status': 0}, {'tag': s._param_tag()}]},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        self._query('node-compound-4', 'node $and multiple conditions', lambda s: MongoFindQuery('node',
            filter={'$and': [
                {'val_int': {'$gte': 200}},
                {'val_float': {'$lte': 0.7}},
                {'status': {'$lte': 2}},
            ]},
            sort={'val_int': -1},
            limit=s._param_limit()
        ))

        # nested grp fields (dot notation)
        self._query('node-nested-grp-0', 'nodes where embedded grp.depth = 0', lambda s: MongoFindQuery('node',
            filter={'grp.depth': 0},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        self._query('node-nested-grp-1', 'nodes where embedded grp.priority > threshold', lambda s: MongoFindQuery('node',
            filter={'grp.priority': {'$gte': s._param_priority()}},
            sort={'grp.priority': -1},
            limit=s._param_limit()
        ))

        self._query('node-nested-grp-2', 'nodes with specific grp.name', lambda s: MongoFindQuery('node',
            filter={'grp.depth': 1, 'status': 0},
            projection={'node_id': 1, 'grp.grp_id': 1, 'grp.name': 1, '_id': 0},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        # pagination
        self._query('node-page-0', 'node list page 1', lambda s: MongoFindQuery('node',
            filter={'status': 0},
            sort={'node_id': 1},
            limit=20,
            skip=0))

        self._query('node-page-1', 'node list arbitrary page', lambda s: MongoFindQuery('node',
            filter={'status': 0},
            sort={'node_id': 1},
            limit=20,
            skip=s._param_skip(0, 5000)
        ))

        self._query('node-page-2', 'node keyset pagination (val_int, node_id)', lambda s: MongoFindQuery('node',
            filter={'$or': [
                {'val_int': {'$gt': s._param_val_int()}},
                {'val_int': s._param_val_int(), 'node_id': {'$gt': s._param_node_id()}},
            ]},
            sort={'val_int': 1, 'node_id': 1},
            limit=20
        ))

    #endregion
    #region Find group

    def _register_find_grp_queries(self):
        self._query('grp-pk-0', 'grp by id', lambda s: MongoFindQuery('grp',
            filter={'grp_id': s._param_grp_id()}
        ))

        self._query('grp-depth-0', 'root groups (depth=0)', lambda s: MongoFindQuery('grp',
            filter={'depth': 0}, sort={'grp_id': 1}
        ))

        self._query('grp-depth-1', 'child groups (depth=1) sorted by priority', lambda s: MongoFindQuery('grp',
            filter={'depth': 1}, sort={'priority': -1}, limit=s._param_limit()
        ))

        self._query('grp-priority-0', 'groups priority > threshold', lambda s: MongoFindQuery('grp',
            filter={'priority': {'$gte': s._param_priority()}},
            sort={'priority': -1}
        ))

        self._query('grp-priority-1', 'top-10 groups by priority', lambda s: MongoFindQuery('grp',
            filter={}, sort={'priority': -1}, limit=10))

        self._query('grp-parent-0', 'children of a group', lambda s: MongoFindQuery('grp',
            filter={'parent_id': s._param_grp_id()}, sort={'grp_id': 1}
        ))

        self._query('grp-parent-1', 'groups with no parent (root)', lambda s: MongoFindQuery('grp',
            filter={'parent_id': None}, sort={'priority': -1}
        ))

        self._query('grp-ids-0', 'groups by id list', lambda s: MongoFindQuery('grp',
            filter={'grp_id': {'$in': s._param_grp_ids(3, 10)}}
        ))

        self._query('grp-compound-0', 'child groups with priority range', lambda s: MongoFindQuery('grp',
            filter={'depth': 1, 'priority': {'$gte': 0.3, '$lte': 0.9}},
            sort={'priority': -1},
            limit=s._param_limit()
        ))

        self._query('grp-all-0', 'all groups sorted', lambda s: MongoFindQuery('grp',
            filter={}, sort={'depth': 1, 'priority': -1}
        ))

    #endregion
    #region Find log

    def _register_find_log_queries(self):
        self._query('log-pk-0', 'log by id', lambda s: MongoFindQuery('log',
            filter={'log_id': s._param_log_id()}
        ))

        self._query('log-node-0', 'logs for a node', lambda s: MongoFindQuery('log',
            filter={'node_id': s._param_node_id()},
            sort={'occurred_at': -1}
        ))

        self._query('log-node-1', 'logs for a node + kind', lambda s: MongoFindQuery('log',
            filter={'node_id': s._param_node_id(), 'kind': s._param_log_kind()},
            sort={'occurred_at': -1},
            limit=s._param_limit()
        ))

        self._query('log-kind-0', 'logs by kind', lambda s: MongoFindQuery('log',
            filter={'kind': s._param_log_kind()}, sort={'occurred_at': -1}, limit=s._param_limit()
        ))

        self._query('log-kind-1', 'logs by kind with val', lambda s: MongoFindQuery('log',
            filter={'kind': s._param_log_kind(), 'val': {'$ne': None}},
            sort={'val': -1},
            limit=s._param_limit()
        ))

        self._query('log-kind-2', 'logs kind IN list', lambda s: MongoFindQuery('log',
            filter={'kind': {'$in': [0, 2, 4, 6]}},
            sort={'occurred_at': -1},
            limit=s._param_limit()
        ))

        self._query('log-date-0', 'logs after date', lambda s: MongoFindQuery('log',
            filter={'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}},
            sort={'occurred_at': -1},
            limit=s._param_limit()
        ))

        self._query('log-date-1', 'logs in date range', lambda s: MongoFindQuery('log',
            filter={
                'occurred_at': {
                    '$gte': s._param_date_minus_days(180, 365),
                    '$lte': s._param_date_minus_days(0, 90),
                }
            },
            sort={'occurred_at': 1}
        ))

        self._query('log-val-0', 'logs with val >= threshold', lambda s: MongoFindQuery('log',
            filter={'val': {'$gte': s._param_val()}},
            sort={'val': -1},
            limit=s._param_limit()
        ))

        self._query('log-val-1', 'logs with non-null val', lambda s: MongoFindQuery('log',
            filter={'val': {'$ne': None}}, sort={'occurred_at': -1}, limit=s._param_limit()
        ))

        self._query('log-null-0', 'logs with null val (missing measurements)', lambda s: MongoFindQuery('log',
            filter={'val': None}, sort={'occurred_at': -1}, limit=s._param_limit()
        ))

        self._query('log-compound-0', 'logs node + kind + date', lambda s: MongoFindQuery('log',
            filter={
                'node_id': {'$in': s._param_node_ids(5, 20)},
                'kind': s._param_log_kind(),
                'occurred_at': {'$gte': s._param_date_minus_days(90, 365)},
            },
            sort={'occurred_at': -1}
        ))

        self._query('log-nodes-0', 'logs for multiple nodes', lambda s: MongoFindQuery('log',
            filter={'node_id': {'$in': s._param_node_ids(10, 30)}},
            sort={'node_id': 1, 'occurred_at': -1}
        ))

    #endregion
    #region measure

    def _register_find_measure_queries(self):
        self._query('measure-pk-0', 'measure by id', lambda s: MongoFindQuery('measure',
            filter={'measure_id': s._param_measure_id()}
        ))

        self._query('measure-node-0', 'measures for a node', lambda s: MongoFindQuery('measure',
            filter={'node_id': s._param_node_id()},
            sort={'recorded_at': -1}
        ))

        self._query('measure-node-1', 'measures for a node + dim', lambda s: MongoFindQuery('measure',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            sort={'recorded_at': -1}
        ))

        self._query('measure-dim-0', 'measures for dim', lambda s: MongoFindQuery('measure',
            filter={'dim': s._param_dim()},
            sort={'val': -1},
            limit=s._param_limit()
        ))

        self._query('measure-dim-1', 'measures dim + val range', lambda s: MongoFindQuery('measure',
            filter={'dim': s._param_dim(), 'val': {'$gte': 50.0}},
            sort={'val': -1},
            limit=s._param_limit()
        ))

        self._query('measure-val-0', 'measures val >= threshold', lambda s: MongoFindQuery('measure',
            filter={'val': {'$gte': s._param_val()}}, sort={'val': -1}, limit=s._param_limit()
        ))

        self._query('measure-date-0', 'measures recorded after date', lambda s: MongoFindQuery('measure',
            filter={'recorded_at': {'$gte': s._param_date_minus_days(30, 90)}},
            sort={'recorded_at': -1},
            limit=s._param_limit()
        ))

        self._query('measure-date-1', 'measures recorded in date range', lambda s: MongoFindQuery('measure',
            filter={
                'recorded_at': {
                    '$gte': s._param_date_minus_days(365, 730),
                    '$lte': s._param_date_minus_days(0, 90),
                },
                'dim': s._param_dim(),
            },
            sort={'recorded_at': 1}
        ))

        self._query('measure-nodes-0', 'measures for multiple nodes', lambda s: MongoFindQuery('measure',
            filter={'node_id': {'$in': s._param_node_ids(5, 20)}, 'dim': s._param_dim()},
            sort={'recorded_at': -1}
        ))

        self._query('measure-compound-0', 'measure node + dim + date + val', lambda s: MongoFindQuery('measure',
            filter={
                'node_id': s._param_node_id(),
                'dim': {'$in': [0, 1]},
                'recorded_at': {'$gte': s._param_date_minus_days(90, 365)},
                'val': {'$gte': 10.0},
            },
            sort={'recorded_at': -1}
        ))

    #endregion
    #region Find link

    def _register_find_link_queries(self):
        self._query('link-src-0', 'links from a node', lambda s: MongoFindQuery('link',
            filter={'src_id': s._param_node_id()}, sort={'dst_id': 1}
        ))

        self._query('link-src-1', 'links from a node + kind', lambda s: MongoFindQuery('link',
            filter={'src_id': s._param_node_id(), 'kind': s._param_link_kind()},
            sort={'weight': -1}
        ))

        self._query('link-dst-0', 'links to a node', lambda s: MongoFindQuery('link',
            filter={'dst_id': s._param_node_id()}, sort={'src_id': 1}
        ))

        self._query('link-dst-1', 'links to a node + kind', lambda s: MongoFindQuery('link',
            filter={'dst_id': s._param_node_id(), 'kind': s._param_link_kind()},
            sort={'weight': -1}
        ))

        self._query('link-kind-0', 'links by kind', lambda s: MongoFindQuery('link',
            filter={'kind': s._param_link_kind()}, sort={'weight': -1}, limit=s._param_limit()
        ))

        self._query('link-weight-0', 'links with weight > threshold', lambda s: MongoFindQuery('link',
            filter={'weight': {'$gte': s._param_weight()}}, sort={'weight': -1}, limit=s._param_limit()
        ))

        self._query('link-weight-1', 'links weight range + kind', lambda s: MongoFindQuery('link',
            filter={'weight': {'$gte': 0.3, '$lte': 0.8}, 'kind': s._param_link_kind()},
            sort={'weight': -1}
        ))

        self._query('link-date-0', 'links created after date', lambda s: MongoFindQuery('link',
            filter={'created_at': {'$gte': s._param_date_minus_days(90, 365)}},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('link-srcs-0', 'links from multiple nodes', lambda s: MongoFindQuery('link',
            filter={'src_id': {'$in': s._param_node_ids(3, 10)}},
            sort={'weight': -1}
        ))

        self._query('link-dsts-0', 'links to multiple nodes', lambda s: MongoFindQuery('link',
            filter={'dst_id': {'$in': s._param_node_ids(3, 10)}},
            sort={'weight': -1}
        ))

        self._query('link-compound-0', 'high-weight links of given kind', lambda s: MongoFindQuery('link',
            filter={
                'kind': s._param_link_kind(),
                'weight': {'$gte': 0.6},
                'created_at': {'$gte': s._param_date_minus_days(180, 730)},
            },
            sort={'weight': -1},
            limit=s._param_limit()
        ))

    #endregion
    #region Find document

    def _register_find_doc_queries(self):
        self._query('doc-pk-0', 'doc by id', lambda s: MongoFindQuery('doc',
            filter={'doc_id': s._param_doc_id()}
        ))

        self._query('doc-node-0', 'doc for a node', lambda s: MongoFindQuery('doc',
            filter={'node_id': s._param_node_id()}
        ))

        self._query('doc-meta-lang-0', 'docs by language', lambda s: MongoFindQuery('doc',
            filter={'meta.lang': s._param_choice('lang', ['en', 'de', 'fr', 'es', 'pt', 'ja', 'zh'])},
            projection={'doc_id': 1, 'node_id': 1, 'meta.lang': 1, 'meta.score': 1, '_id': 0},
            sort={'meta.score': -1},
            limit=s._param_limit()
        ))

        self._query('doc-meta-version-0', 'docs with meta version', lambda s: MongoFindQuery('doc',
            filter={'meta.version': s._param_int('version', 1, 5)},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('doc-meta-score-0', 'docs with high score', lambda s: MongoFindQuery('doc',
            filter={'meta.score': {'$gte': 8.0}},
            sort={'meta.score': -1},
            limit=s._param_limit()
        ))

        self._query('doc-meta-score-1', 'docs score in range + lang', lambda s: MongoFindQuery('doc',
            filter={
                'meta.score': {'$gte': 5.0, '$lte': 9.0},
                'meta.lang': s._param_choice('lang', ['en', 'de', 'fr']),
            },
            sort={'meta.score': -1},
            limit=s._param_limit()
        ))

        self._query('doc-date-0', 'docs created after date', lambda s: MongoFindQuery('doc',
            filter={'created_at': {'$gte': s._param_date_minus_days(180, 730)}},
            sort={'created_at': -1},
            limit=s._param_limit()
        ))

        self._query('doc-nodes-0', 'docs for multiple nodes', lambda s: MongoFindQuery('doc',
            filter={'node_id': {'$in': s._param_node_ids(5, 20)}},
            sort={'node_id': 1}
        ))

        self._query('doc-compound-0', 'docs lang + score + date', lambda s: MongoFindQuery('doc',
            filter={
                'meta.lang': s._param_choice('lang', ['en', 'de', 'fr']),
                'meta.score': {'$gte': 6.0},
                'created_at': {'$gte': s._param_date_minus_days(365, 730)},
            },
            sort={'meta.score': -1},
            limit=s._param_limit()
        ))

    #endregion
    #region array/nested

    def _register_node_array_queries(self):
        # $unwind logs
        self._query('node-unwind-log-0', 'node log events count per node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$project': {'node_id': 1, 'log_count': {'$size': '$logs'}}},
            {'$sort': {'log_count': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-unwind-log-1', 'unwind logs, group by kind', lambda s: MongoAggregateQuery('node', [
            {'$match': {'status': s._param_status()}},
            {'$unwind': '$logs'},
            {'$group': {'_id': '$logs.kind', 'count': {'$sum': 1}, 'avg_val': {'$avg': '$logs.val'}}},
            {'$sort': {'count': -1}},
        ]))

        self._query('node-unwind-log-2', 'unwind logs, filter by kind, group by node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'grp_id': s._param_grp_id()}},
            {'$unwind': '$logs'},
            {'$match': {'logs.kind': s._param_log_kind()}},
            {'$group': {
                '_id': '$node_id',
                'tag': {'$first': '$tag'},
                'log_count': {'$sum': 1},
                'avg_val': {'$avg': '$logs.val'},
            }},
            {'$sort': {'log_count': -1}},
        ]))

        self._query('node-unwind-log-3', 'unwind logs after date, count', lambda s: MongoAggregateQuery('node', [
            {'$unwind': '$logs'},
            {'$match': {'logs.occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$group': {'_id': '$node_id', 'recent_logs': {'$sum': 1}}},
            {'$sort': {'recent_logs': -1}},
            {'$limit': 50},
        ]))

        self._query('node-unwind-log-4', 'unwind logs, distinct kinds per node', lambda s: MongoAggregateQuery('node', [
            {'$unwind': '$logs'},
            {'$group': {'_id': '$node_id', 'kinds': {'$addToSet': '$logs.kind'}}},
            {'$addFields': {'n_kinds': {'$size': '$kinds'}}},
            {'$match': {'n_kinds': {'$gte': 3}}},
            {'$sort': {'n_kinds': -1}},
        ]))

        # $unwind measures
        self._query('node-unwind-measure-0', 'unwind measures, group by dim', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$unwind': '$measures'},
            {'$group': {
                '_id': '$measures.dim',
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$measures.val'},
                'max_val': {'$max': '$measures.val'},
            }},
            {'$sort': {'avg_val': -1}},
        ]))

        self._query('node-unwind-measure-1', 'unwind measures dim=0, top nodes by avg', lambda s: MongoAggregateQuery('node', [
            {'$unwind': '$measures'},
            {'$match': {'measures.dim': 0}},
            {'$group': {
                '_id': '$node_id',
                'tag': {'$first': '$tag'},
                'avg_val': {'$avg': '$measures.val'},
                'max_val': {'$max': '$measures.val'},
            }},
            {'$sort': {'avg_val': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-unwind-measure-2', 'unwind measures, recent date window', lambda s: MongoAggregateQuery('node', [
            {'$unwind': '$measures'},
            {'$match': {'measures.recorded_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$group': {
                '_id': {'node_id': '$node_id', 'dim': '$measures.dim'},
                'avg_val': {'$avg': '$measures.val'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'avg_val': -1}},
        ]))

        # $unwind out_links
        self._query('node-unwind-link-0', 'unwind out_links, out-degree per node', lambda s: MongoAggregateQuery('node', [
            {'$project': {'node_id': 1, 'out_degree': {'$size': '$out_links'}}},
            {'$match': {'out_degree': {'$gt': 0}}},
            {'$sort': {'out_degree': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-unwind-link-1', 'unwind out_links, group by kind', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$unwind': '$out_links'},
            {'$group': {
                '_id': '$out_links.kind',
                'count': {'$sum': 1},
                'avg_weight': {'$avg': '$out_links.weight'},
            }},
            {'$sort': {'count': -1}},
        ]))

        self._query('node-unwind-link-2', 'nodes with heavy out_links (weight > thr)', lambda s: MongoAggregateQuery('node', [
            {'$unwind': '$out_links'},
            {'$match': {'out_links.weight': {'$gte': s._param_weight()}}},
            {'$group': {
                '_id': '$node_id',
                'tag': {'$first': '$tag'},
                'heavy_links': {'$sum': 1},
                'max_weight': {'$max': '$out_links.weight'},
            }},
            {'$sort': {'heavy_links': -1}},
            {'$limit': s._param_limit()},
        ]))

        # $elemMatch
        self._query('node-elem-log-0', 'nodes with at least one log of given kind', lambda s: MongoFindQuery('node',
            filter={'logs': {'$elemMatch': {'kind': s._param_log_kind()}}},
            projection={'node_id': 1, 'tag': 1, '_id': 0},
            limit=s._param_limit()
        ))

        self._query('node-elem-log-1', 'nodes with high-value log event', lambda s: MongoFindQuery('node',
            filter={'logs': {'$elemMatch': {'kind': s._param_log_kind(), 'val': {'$gte': 80.0}}}},
            projection={'node_id': 1, 'tag': 1, '_id': 0},
            limit=s._param_limit()
        ))

        self._query('node-elem-log-2', 'nodes with recent log event', lambda s: MongoFindQuery('node',
            filter={'logs': {'$elemMatch': {'occurred_at': {'$gte': s._param_date_minus_days(7, 30)}}}},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        self._query('node-elem-measure-0', 'nodes with dim=3 measure > threshold', lambda s: MongoFindQuery('node',
            filter={'measures': {'$elemMatch': {'dim': 3, 'val': {'$gte': 5000.0}}}},
            projection={'node_id': 1, 'tag': 1, '_id': 0},
            limit=s._param_limit()
        ))

        self._query('node-elem-measure-1', 'nodes with recent measure of given dim', lambda s: MongoFindQuery('node',
            filter={'measures': {'$elemMatch': {
                'dim': s._param_dim(),
                'recorded_at': {'$gte': s._param_date_minus_days(30, 90)},
            }}},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        self._query('node-elem-link-0', 'nodes with heavy outgoing link of given kind', lambda s: MongoFindQuery('node',
            filter={'out_links': {'$elemMatch': {'kind': s._param_link_kind(), 'weight': {'$gte': 0.7}}}},
            projection={'node_id': 1, 'tag': 1, 'out_links': 1, '_id': 0},
            limit=s._param_limit()
        ))

        # $size
        self._query('node-size-log-0', 'nodes with many log events', lambda s: MongoAggregateQuery('node', [
            {'$addFields': {'log_count': {'$size': '$logs'}}},
            {'$match': {'log_count': {'$gte': 10}}},
            {'$sort': {'log_count': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-size-measure-0', 'nodes with many measures', lambda s: MongoAggregateQuery('node', [
            {'$addFields': {'measure_count': {'$size': '$measures'}}},
            {'$match': {'measure_count': {'$gte': 5}}},
            {'$sort': {'measure_count': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-size-link-0', 'hub nodes (high out-degree)', lambda s: MongoAggregateQuery('node', [
            {'$addFields': {'out_degree': {'$size': '$out_links'}}},
            {'$sort': {'out_degree': -1}},
            {'$limit': s._param_limit()},
            {'$project': {'node_id': 1, 'tag': 1, 'out_degree': 1, '_id': 0}},
        ]))

        # $filter on arrays
        self._query('node-filter-log-0', 'filter logs by kind within node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$project': {
                'node_id': 1,
                'filtered_logs': {
                    '$filter': {
                        'input': '$logs',
                        'as': 'l',
                        'cond': {'$eq': ['$$l.kind', s._param_log_kind()]},
                    }
                },
            }},
        ]))

        self._query('node-filter-measure-0', 'filter measures by dim and val', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$project': {
                'node_id': 1,
                'dim_measures': {
                    '$filter': {
                        'input': '$measures',
                        'as': 'm',
                        'cond': {'$and': [
                            {'$eq': ['$$m.dim', s._param_dim()]},
                            {'$gte': ['$$m.val', 50.0]},
                        ]},
                    }
                },
            }},
        ]))

        self._query('node-filter-link-0', 'filter out_links by kind', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True, 'node_id': {'$in': s._param_node_ids(5, 20)}}},
            {'$project': {
                'node_id': 1,
                'typed_links': {
                    '$filter': {
                        'input': '$out_links',
                        'as': 'lk',
                        'cond': {'$eq': ['$$lk.kind', s._param_link_kind()]},
                    }
                },
            }},
        ]))

        # $reduce / $map
        self._query('node-reduce-log-0', 'sum of log vals per node via $reduce', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True, 'node_id': {'$in': s._param_node_ids(10, 30)}}},
            {'$project': {
                'node_id': 1,
                'total_log_val': {
                    '$reduce': {
                        'input': {'$filter': {
                            'input': '$logs',
                            'as': 'l',
                            'cond': {'$ne': ['$$l.val', None]},
                        }},
                        'initialValue': 0,
                        'in': {'$add': ['$$value', '$$this.val']},
                    }
                },
            }},
            {'$sort': {'total_log_val': -1}},
        ]))

        self._query('node-map-link-0', 'map out_links to dst_id list', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': {'$in': s._param_node_ids(5, 10)}}},
            {'$project': {
                'node_id': 1,
                'dst_ids': {'$map': {'input': '$out_links', 'as': 'lk', 'in': '$$lk.dst_id'}},
            }},
        ]))

        self._query('node-map-measure-0', 'map measures to val list per node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$project': {
                'node_id': 1,
                'measure_vals': {
                    '$map': {
                        'input': {'$filter': {'input': '$measures', 'as': 'm', 'cond': {'$eq': ['$$m.dim', 0]}}},
                        'as': 'm',
                        'in': '$$m.val',
                    }
                },
            }},
        ]))

        # doc array
        self._query('node-doc-0', 'nodes that have a document embedded', lambda s: MongoFindQuery('node',
            filter={'doc': {'$exists': True, '$ne': []}},
            projection={'node_id': 1, 'tag': 1, '_id': 0},
            sort={'node_id': 1},
            limit=s._param_limit()
        ))

        self._query('node-doc-1', 'nodes with doc where lang = X', lambda s: MongoAggregateQuery('node', [
            {'$match': {'doc': {'$ne': []}}},
            {'$unwind': '$doc'},
            {'$match': {'doc.meta.lang': s._param_choice('lang', ['en', 'de', 'fr', 'es'])}},
            {'$project': {'node_id': 1, 'tag': 1, 'doc.doc_id': 1, 'doc.meta': 1, '_id': 0}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-doc-2', 'nodes with high-score doc', lambda s: MongoAggregateQuery('node', [
            {'$match': {'doc': {'$ne': []}}},
            {'$unwind': '$doc'},
            {'$match': {'doc.meta.score': {'$gte': 8.0}}},
            {'$sort': {'doc.meta.score': -1}},
            {'$limit': s._param_limit()},
        ]))

    #endregion
    #region nested compound filters

    def _register_node_nested_filter_queries(self):
        self._query('node-nf-0', 'active hub nodes (many logs + many links)', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$addFields': {
                'log_count': {'$size': '$logs'},
                'out_degree': {'$size': '$out_links'},
            }},
            {'$match': {'log_count': {'$gte': 5}, 'out_degree': {'$gte': 3}}},
            {'$sort': {'out_degree': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-nf-1', 'nodes with grp.depth=0 and log events of kind X', lambda s: MongoAggregateQuery('node', [
            {'$match': {
                'grp.depth': 0,
                'logs': {'$elemMatch': {'kind': s._param_log_kind()}},
            }},
            {'$project': {'node_id': 1, 'tag': 1, 'grp.name': 1, '_id': 0}},
            {'$sort': {'node_id': 1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('node-nf-2', 'nodes status<=1, val_int>=500, grp.priority>0.7', lambda s: MongoFindQuery('node',
            filter={
                'status': {'$lte': 1},
                'val_int': {'$gte': 500},
                'grp.priority': {'$gte': 0.7},
            },
            sort={'val_int': -1},
            limit=s._param_limit()
        ))

        self._query('node-nf-3', 'total weight of all out_links per node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$addFields': {
                'total_weight': {'$sum': '$out_links.weight'},
                'out_degree': {'$size': '$out_links'},
            }},
            {'$match': {'total_weight': {'$gte': 1.0}}},
            {'$sort': {'total_weight': -1}},
            {'$limit': s._param_limit()},
            {'$project': {'node_id': 1, 'tag': 1, 'out_degree': 1, 'total_weight': 1, '_id': 0}},
        ]))

        self._query('node-nf-4', 'avg log val + measure dim summary per node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': {'$in': s._param_node_ids(20, 50)}}},
            {'$project': {
                'node_id': 1,
                'tag': 1,
                'avg_log_val': {'$avg': '$logs.val'},
                'measure_count': {'$size': '$measures'},
                'avg_measure_val': {'$avg': '$measures.val'},
            }},
            {'$sort': {'avg_log_val': -1}},
        ]))

        self._query('node-nf-5', 'nodes with both doc and at least one heavy link', lambda s: MongoAggregateQuery('node', [
            {'$match': {
                'doc': {'$ne': []},
                'out_links': {'$elemMatch': {'weight': {'$gte': 0.8}}},
            }},
            {'$project': {'node_id': 1, 'tag': 1, 'grp.name': 1, '_id': 0}},
            {'$sort': {'node_id': 1}},
            {'$limit': s._param_limit()},
        ]))

    #endregion
    #region Aggregation (single)

    def _register_aggregation_queries(self):
        # count queries
        self._query('agg-count-0', 'count nodes by status', lambda s: MongoAggregateQuery('node', [
            {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
        ]))

        self._query('agg-count-1', 'count nodes by grp', lambda s: MongoAggregateQuery('node', [
            {'$group': {'_id': '$grp_id', 'count': {'$sum': 1}, 'active': {'$sum': {'$cond': ['$is_active', 1, 0]}}}},
            {'$sort': {'count': -1}},
            {'$limit': 20},
        ]))

        self._query('agg-count-2', 'count nodes by tag (top tags)', lambda s: MongoAggregateQuery('node', [
            {'$group': {'_id': '$tag', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 20},
        ]))

        self._query('agg-count-3', 'count logs by kind', lambda s: MongoAggregateQuery('log', [
            {'$group': {'_id': '$kind', 'count': {'$sum': 1}, 'null_val_count': {'$sum': {'$cond': [{'$eq': ['$val', None]}, 1, 0]}}}},
            {'$sort': {'count': -1}},
        ]))

        self._query('agg-count-4', 'count measures by dim', lambda s: MongoAggregateQuery('measure', [
            {'$group': {'_id': '$dim', 'count': {'$sum': 1}, 'avg_val': {'$avg': '$val'}, 'max_val': {'$max': '$val'}}},
            {'$sort': {'_id': 1}},
        ]))

        self._query('agg-count-5', 'count links by kind', lambda s: MongoAggregateQuery('link', [
            {'$group': {'_id': '$kind', 'count': {'$sum': 1}, 'avg_weight': {'$avg': '$weight'}}},
            {'$sort': {'count': -1}},
        ]))

        # sum / avg / min / max
        self._query('agg-stat-0', 'node val_int stats by grp', lambda s: MongoAggregateQuery('node', [
            {'$group': {
                '_id': '$grp_id',
                'cnt': {'$sum': 1},
                'avg_vi': {'$avg': '$val_int'},
                'min_vi': {'$min': '$val_int'},
                'max_vi': {'$max': '$val_int'},
            }},
            {'$sort': {'avg_vi': -1}},
            {'$limit': 20},
        ]))

        self._query('agg-stat-1', 'node val_float stats by status', lambda s: MongoAggregateQuery('node', [
            {'$group': {
                '_id': '$status',
                'cnt': {'$sum': 1},
                'avg_vf': {'$avg': '$val_float'},
                'min_vf': {'$min': '$val_float'},
                'max_vf': {'$max': '$val_float'},
                'stddev': {'$stdDevPop': '$val_float'},
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('agg-stat-2', 'measure val stats by dim', lambda s: MongoAggregateQuery('measure', [
            {'$group': {
                '_id': '$dim',
                'cnt': {'$sum': 1},
                'avg': {'$avg': '$val'},
                'min': {'$min': '$val'},
                'max': {'$max': '$val'},
                'stddev': {'$stdDevSamp': '$val'},
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('agg-stat-3', 'log val stats by kind', lambda s: MongoAggregateQuery('log', [
            {'$match': {'val': {'$ne': None}}},
            {'$group': {
                '_id': '$kind',
                'cnt': {'$sum': 1},
                'avg': {'$avg': '$val'},
                'min': {'$min': '$val'},
                'max': {'$max': '$val'},
            }},
            {'$sort': {'avg': -1}},
        ]))

        # $bucket / $bucketAuto
        self._query('agg-bucket-0', 'node val_int histogram (10 buckets)', lambda s: MongoAggregateQuery('node', [
            {'$bucket': {
                'groupBy': '$val_int',
                'boundaries': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1001],
                'default': 'Other',
                'output': {'count': {'$sum': 1}, 'avg_vf': {'$avg': '$val_float'}},
            }},
        ]))

        self._query('agg-bucket-1', 'node val_float auto-bucket', lambda s: MongoAggregateQuery('node', [
            {'$bucketAuto': {
                'groupBy': '$val_float',
                'buckets': 8,
                'output': {'count': {'$sum': 1}},
            }},
        ]))

        self._query('agg-bucket-2', 'measure val buckets by dim', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'dim': s._param_dim()}},
            {'$bucketAuto': {
                'groupBy': '$val',
                'buckets': 10,
                'output': {'count': {'$sum': 1}, 'avg_val': {'$avg': '$val'}},
            }},
        ]))

        # $sortByCount
        self._query('agg-sortbycount-0', 'most frequent tags', lambda s: MongoAggregateQuery('node', [
            {'$sortByCount': '$tag'},
            {'$limit': 20}
        ]))

        self._query('agg-sortbycount-1', 'most active grp ids', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$sortByCount': '$grp_id'},
            {'$limit': 20}
        ]))

        self._query('agg-sortbycount-2', 'most common log kinds', lambda s: MongoAggregateQuery('log', [
            {'$sortByCount': '$kind'}
        ]))

        # top-K
        self._query('agg-topk-0', 'top nodes by val_int', lambda s: MongoAggregateQuery('node', [
            {'$match': {'status': 0}},
            {'$sort': {'val_int': -1}},
            {'$limit': 20},
            {'$project': {'node_id': 1, 'tag': 1, 'val_int': 1, '_id': 0}},
        ]))

        self._query('agg-topk-1', 'top links by weight per kind', lambda s: MongoAggregateQuery('link', [
            {'$sort': {'kind': 1, 'weight': -1}},
            {'$group': {
                '_id': '$kind',
                'top_link': {'$first': {'src_id': '$src_id', 'dst_id': '$dst_id', 'weight': '$weight'}},
            }},
        ]))

        self._query('agg-topk-2', 'top measures per dim (max val)', lambda s: MongoAggregateQuery('measure', [
            {'$sort': {'dim': 1, 'val': -1}},
            {'$group': {
                '_id': '$dim',
                'top_measure': {'$first': {'node_id': '$node_id', 'val': '$val', 'recorded_at': '$recorded_at'}},
            }},
        ]))

        # date truncation / time-series
        self._query('agg-timeseries-0', 'log events per day', lambda s: MongoAggregateQuery('log', [
            {'$match': {'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$group': {
                '_id': {'$dateTrunc': {'date': '$occurred_at', 'unit': 'day'}},
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$val'},
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('agg-timeseries-1', 'measure val per week', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'dim': s._param_dim(), 'recorded_at': {'$gte': s._param_date_minus_days(90, 365)}}},
            {'$group': {
                '_id': {'$dateTrunc': {'date': '$recorded_at', 'unit': 'week'}},
                'avg_val': {'$avg': '$val'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('agg-timeseries-2', 'node creation by month', lambda s: MongoAggregateQuery('node', [
            {'$group': {
                '_id': {'$dateTrunc': {'date': '$created_at', 'unit': 'month'}},
                'count': {'$sum': 1},
                'active': {'$sum': {'$cond': ['$is_active', 1, 0]}},
            }},
            {'$sort': {'_id': 1}},
        ]))

        # multi-field group
        self._query('agg-multi-group-0', 'node count by grp+status', lambda s: MongoAggregateQuery('node', [
            {'$group': {
                '_id': {'grp_id': '$grp_id', 'status': '$status'},
                'count': {'$sum': 1},
                'avg_vi': {'$avg': '$val_int'},
            }},
            {'$sort': {'count': -1}},
            {'$limit': 50},
        ]))

        self._query('agg-multi-group-1', 'measure count by node+dim', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'node_id': {'$in': s._param_node_ids(10, 30)}}},
            {'$group': {
                '_id': {'node_id': '$node_id', 'dim': '$dim'},
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$val'},
            }},
            {'$sort': {'avg_val': -1}},
        ]))

        # $addFields + computed
        self._query('agg-computed-0', 'node val_int + val_float combined score', lambda s: MongoAggregateQuery('node', [
            {'$addFields': {'score': {'$add': ['$val_int', {'$multiply': ['$val_float', 100]}]}}},
            {'$sort': {'score': -1}},
            {'$limit': s._param_limit()},
            {'$project': {'node_id': 1, 'tag': 1, 'score': 1, '_id': 0}},
        ]))

        self._query('agg-computed-1', 'log age in days', lambda s: MongoAggregateQuery('log', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$addFields': {'age_ms': {'$subtract': ['$$NOW', '$occurred_at']}}},
            {'$addFields': {'age_days': {'$divide': ['$age_ms', 86400000]}}},
            {'$sort': {'age_days': 1}},
        ]))

        # $project with conditional
        self._query('agg-project-0', 'node status label', lambda s: MongoAggregateQuery('node', [
            {'$match': {'tag': s._param_tag()}},
            {'$project': {
                'node_id': 1,
                'tag': 1,
                'status_label': {'$switch': {
                    'branches': [
                        {'case': {'$eq': ['$status', 0]}, 'then': 'active'},
                        {'case': {'$eq': ['$status', 1]}, 'then': 'inactive'},
                        {'case': {'$eq': ['$status', 2]}, 'then': 'pending'},
                        {'case': {'$eq': ['$status', 3]}, 'then': 'banned'},
                    ],
                    'default': 'deleted',
                }},
                '_id': 0,
            }},
        ]))

        self._query('agg-project-1', 'node is_heavy computed flag', lambda s: MongoAggregateQuery('node', [
            {'$match': {'grp_id': s._param_grp_id()}},
            {'$addFields': {
                'is_heavy': {'$gte': ['$val_int', 500]},
                'link_count': {'$size': '$out_links'},
            }},
            {'$project': {'node_id': 1, 'tag': 1, 'is_heavy': 1, 'link_count': 1, '_id': 0}},
            {'$sort': {'link_count': -1}},
        ]))

        # $match with $expr
        self._query('agg-expr-0', 'nodes where val_int > val_float * 700', lambda s: MongoAggregateQuery('node', [
            {'$match': {'$expr': {'$gt': ['$val_int', {'$multiply': ['$val_float', 700]}]}}},
            {'$sort': {'val_int': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('agg-expr-1', 'log events where val > avg(val) proxy', lambda s: MongoAggregateQuery('log', [
            {'$match': {'val': {'$ne': None}}},
            {'$addFields': {'high': {'$gt': ['$val', 50.0]}}},
            {'$match': {'high': True}},
            {'$sort': {'val': -1}},
            {'$limit': s._param_limit()},
        ]))

    #endregion
    #region $lookup (cross)

    def _register_lookup_queries(self):
        # log -> node
        self._query('lookup-log-node-0', 'log events with node info', lambda s: MongoAggregateQuery('log', [
            {'$match': {'kind': s._param_log_kind()}},
            {'$lookup': {
                'from': 'node',
                'localField': 'node_id',
                'foreignField': 'node_id',
                'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$project': {
                'log_id': 1, 'kind': 1, 'val': 1, 'occurred_at': 1,
                'node.tag': 1, 'node.status': 1, 'node.grp_id': 1, '_id': 0,
            }},
            {'$sort': {'occurred_at': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('lookup-log-node-1', 'log events for active nodes only', lambda s: MongoAggregateQuery('log', [
            {'$match': {'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$lookup': {
                'from': 'node',
                'localField': 'node_id',
                'foreignField': 'node_id',
                'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$match': {'node.is_active': True}},
            {'$group': {
                '_id': '$kind',
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$val'},
            }},
            {'$sort': {'count': -1}},
        ]))

        self._query('lookup-log-node-2', 'average log val per grp', lambda s: MongoAggregateQuery('log', [
            {'$match': {'val': {'$ne': None}, 'kind': s._param_log_kind()}},
            {'$lookup': {
                'from': 'node',
                'localField': 'node_id',
                'foreignField': 'node_id',
                'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$group': {
                '_id': '$node.grp_id',
                'avg_val': {'$avg': '$val'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'avg_val': -1}},
            {'$limit': 20},
        ]))

        # measure -> node
        self._query('lookup-measure-node-0', 'measure val with node tag', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'dim': s._param_dim(), 'val': {'$gte': s._param_val()}}},
            {'$lookup': {
                'from': 'node',
                'localField': 'node_id',
                'foreignField': 'node_id',
                'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$project': {
                'measure_id': 1, 'dim': 1, 'val': 1, 'recorded_at': 1,
                'node.tag': 1, 'node.status': 1, '_id': 0,
            }},
            {'$sort': {'val': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('lookup-measure-node-1', 'measure stats per grp via node join', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'dim': s._param_dim()}},
            {'$lookup': {
                'from': 'node',
                'localField': 'node_id',
                'foreignField': 'node_id',
                'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$group': {
                '_id': '$node.grp_id',
                'avg_val': {'$avg': '$val'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'avg_val': -1}},
            {'$limit': 20},
        ]))

        # link -> node (src/dst)
        self._query('lookup-link-src-0', 'links with src node info', lambda s: MongoAggregateQuery('link', [
            {'$match': {'kind': s._param_link_kind(), 'weight': {'$gte': s._param_weight()}}},
            {'$lookup': {
                'from': 'node',
                'localField': 'src_id',
                'foreignField': 'node_id',
                'as': 'src_node',
            }},
            {'$unwind': '$src_node'},
            {'$project': {
                'src_id': 1, 'dst_id': 1, 'kind': 1, 'weight': 1,
                'src_node.tag': 1, 'src_node.status': 1, '_id': 0,
            }},
            {'$sort': {'weight': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('lookup-link-both-0', 'links with both src + dst node info', lambda s: MongoAggregateQuery('link', [
            {'$match': {'kind': s._param_link_kind()}},
            {'$lookup': {
                'from': 'node',
                'localField': 'src_id',
                'foreignField': 'node_id',
                'as': 'src_node',
            }},
            {'$unwind': '$src_node'},
            {'$lookup': {
                'from': 'node',
                'localField': 'dst_id',
                'foreignField': 'node_id',
                'as': 'dst_node',
            }},
            {'$unwind': '$dst_node'},
            {'$project': {
                'kind': 1, 'weight': 1,
                'src_tag': '$src_node.tag', 'src_status': '$src_node.status',
                'dst_tag': '$dst_node.tag', 'dst_status': '$dst_node.status',
                '_id': 0,
            }},
            {'$sort': {'weight': -1}},
            {'$limit': s._param_limit()},
        ]))

        # node -> grp (explicit, no embedding)
        self._query('lookup-node-grp-0', 'node list with grp name via $lookup', lambda s: MongoAggregateQuery('node', [
            {'$match': {'status': s._param_status(), 'tag': s._param_tag()}},
            {'$lookup': {
                'from': 'grp',
                'localField': 'grp_id',
                'foreignField': 'grp_id',
                'as': 'group',
            }},
            {'$unwind': '$group'},
            {'$project': {
                'node_id': 1, 'tag': 1,
                'group.name': 1, 'group.depth': 1, 'group.priority': 1,
                '_id': 0,
            }},
            {'$sort': {'node_id': 1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('lookup-node-grp-1', 'count nodes per grp.depth via join', lambda s: MongoAggregateQuery('node', [
            {'$lookup': {
                'from': 'grp',
                'localField': 'grp_id',
                'foreignField': 'grp_id',
                'as': 'group',
            }},
            {'$unwind': '$group'},
            {'$group': {
                '_id': '$group.depth',
                'count': {'$sum': 1},
                'active': {'$sum': {'$cond': ['$is_active', 1, 0]}},
            }},
            {'$sort': {'_id': 1}},
        ]))

        # pipeline $lookup (correlated subquery-style)
        self._query('lookup-pipeline-0', 'nodes with log count via pipeline lookup', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True, 'grp_id': s._param_grp_id()}},
            {'$lookup': {
                'from': 'log',
                'let': {'nid': '$node_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$node_id', '$$nid']}}},
                    {'$count': 'count'},
                ],
                'as': 'log_stats',
            }},
            {'$addFields': {
                'log_count': {'$ifNull': [{'$arrayElemAt': ['$log_stats.count', 0]}, 0]},
            }},
            {'$sort': {'log_count': -1}},
            {'$project': {'node_id': 1, 'tag': 1, 'log_count': 1, '_id': 0}},
        ]))

        self._query('lookup-pipeline-1', 'nodes with recent measure avg via pipeline lookup', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True, 'node_id': {'$in': s._param_node_ids(20, 50)}}},
            {'$lookup': {
                'from': 'measure',
                'let': {'nid': '$node_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$and': [
                        {'$eq': ['$node_id', '$$nid']},
                        {'$gte': ['$recorded_at', s._param_date_minus_days(30, 90)]},
                    ]}}},
                    {'$group': {'_id': None, 'avg_val': {'$avg': '$val'}, 'count': {'$sum': 1}}},
                ],
                'as': 'recent_measures',
            }},
            {'$addFields': {
                'recent_avg': {'$ifNull': [{'$arrayElemAt': ['$recent_measures.avg_val', 0]}, None]},
                'recent_count': {'$ifNull': [{'$arrayElemAt': ['$recent_measures.count', 0]}, 0]},
            }},
            {'$match': {'recent_count': {'$gte': 1}}},
            {'$sort': {'recent_avg': -1}},
            {'$project': {'node_id': 1, 'tag': 1, 'recent_avg': 1, 'recent_count': 1, '_id': 0}},
        ]))

        self._query('lookup-pipeline-2', 'grp with active-node count via pipeline lookup', lambda s: MongoAggregateQuery('grp', [
            {'$lookup': {
                'from': 'node',
                'let': {'gid': '$grp_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$and': [
                        {'$eq': ['$grp_id', '$$gid']},
                        {'$eq': ['$is_active', True]},
                    ]}}},
                    {'$count': 'count'},
                ],
                'as': 'active_nodes',
            }},
            {'$addFields': {
                'active_count': {'$ifNull': [{'$arrayElemAt': ['$active_nodes.count', 0]}, 0]},
            }},
            {'$sort': {'active_count': -1}},
            {'$project': {'grp_id': 1, 'name': 1, 'depth': 1, 'active_count': 1, '_id': 0}},
        ]))

        self._query('lookup-pipeline-3', 'doc with node status via pipeline lookup', lambda s: MongoAggregateQuery('doc', [
            {'$match': {'meta.lang': s._param_choice('lang', ['en', 'de', 'fr'])}},
            {'$lookup': {
                'from': 'node',
                'let': {'nid': '$node_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$node_id', '$$nid']}}},
                    {'$project': {'tag': 1, 'status': 1, 'grp_id': 1, '_id': 0}},
                ],
                'as': 'node',
            }},
            {'$unwind': {'path': '$node', 'preserveNullAndEmptyArrays': True}},
            {'$project': {
                'doc_id': 1, 'node_id': 1,
                'meta.lang': 1, 'meta.score': 1,
                'node.tag': 1, 'node.status': 1,
                '_id': 0,
            }},
            {'$sort': {'meta.score': -1}},
            {'$limit': s._param_limit()},
        ]))

    #endregion
    #region $graphLookup

    def _register_graphlookup_queries(self):
        # Traverse grp hierarchy upward (child -> root)
        self._query('graph-grp-0', 'grp ancestors of a group (upward traversal)', lambda s: MongoAggregateQuery('grp', [
            {'$match': {'grp_id': s._param_grp_id()}},
            {'$graphLookup': {
                'from': 'grp',
                'startWith': '$parent_id',
                'connectFromField': 'parent_id',
                'connectToField': 'grp_id',
                'as': 'ancestors',
                'maxDepth': 10,
            }},
            {'$project': {'grp_id': 1, 'name': 1, 'depth': 1, 'ancestors.grp_id': 1, 'ancestors.name': 1, '_id': 0}},
        ]))

        # Traverse grp hierarchy downward (root -> children)
        self._query('graph-grp-1', 'grp subtree from root (downward traversal)', lambda s: MongoAggregateQuery('grp', [
            {'$match': {'depth': 0, 'grp_id': s._param_grp_id()}},
            {'$graphLookup': {
                'from': 'grp',
                'startWith': '$grp_id',
                'connectFromField': 'grp_id',
                'connectToField': 'parent_id',
                'as': 'descendants',
                'maxDepth': 10,
            }},
            {'$project': {'grp_id': 1, 'name': 1, 'descendants.grp_id': 1, 'descendants.name': 1, '_id': 0}},
        ]))

        # Traverse link graph
        self._query('graph-link-0', 'link graph reachable from a node (depth 2)', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$graphLookup': {
                'from': 'link',
                'startWith': '$node_id',
                'connectFromField': 'dst_id',
                'connectToField': 'src_id',
                'as': 'reachable',
                'maxDepth': 2,
                'restrictSearchWithMatch': {'kind': s._param_link_kind()},
            }},
            {'$project': {'node_id': 1, 'tag': 1, 'reachable.dst_id': 1, 'reachable.kind': 1, 'reachable.weight': 1, '_id': 0}},
        ]))

        self._query('graph-link-1', 'link graph predecessors of a node', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$graphLookup': {
                'from': 'link',
                'startWith': '$node_id',
                'connectFromField': 'src_id',
                'connectToField': 'dst_id',
                'as': 'predecessors',
                'maxDepth': 2,
            }},
            {'$project': {'node_id': 1, 'predecessors.src_id': 1, 'predecessors.weight': 1, '_id': 0}},
        ]))

        self._query('graph-link-2', 'all nodes reachable from hub nodes', lambda s: MongoAggregateQuery('node', [
            {'$addFields': {'out_degree': {'$size': '$out_links'}}},
            {'$sort': {'out_degree': -1}},
            {'$limit': 5},
            {'$graphLookup': {
                'from': 'link',
                'startWith': '$node_id',
                'connectFromField': 'dst_id',
                'connectToField': 'src_id',
                'as': 'reachable',
                'maxDepth': 3,
            }},
            {'$project': {'node_id': 1, 'tag': 1, 'out_degree': 1, 'reachable_count': {'$size': '$reachable'}, '_id': 0}},
        ]))

        self._query('graph-grp-2', 'nodes in entire subtree of a group', lambda s: MongoAggregateQuery('grp', [
            {'$match': {'depth': 0}},
            {'$graphLookup': {
                'from': 'grp',
                'startWith': '$grp_id',
                'connectFromField': 'grp_id',
                'connectToField': 'parent_id',
                'as': 'sub_grps',
                'maxDepth': 5,
            }},
            {'$project': {'grp_id': 1, 'name': 1, 'sub_grp_ids': '$sub_grps.grp_id', '_id': 0}},
            {'$lookup': {
                'from': 'node',
                'localField': 'sub_grp_ids',
                'foreignField': 'grp_id',
                'as': 'nodes_in_subtree',
            }},
            {'$addFields': {'subtree_node_count': {'$size': '$nodes_in_subtree'}}},
            {'$sort': {'subtree_node_count': -1}},
            {'$limit': 10},
            {'$project': {'grp_id': 1, 'name': 1, 'subtree_node_count': 1, '_id': 0}},
        ]))

    #endregion
    #region $setWindowFields (window)

    def _register_window_queries(self):
        self._query('window-rank-0', 'rank nodes by val_int within grp', lambda s: MongoAggregateQuery('node', [
            {'$setWindowFields': {
                'partitionBy': '$grp_id',
                'sortBy': {'val_int': -1},
                'output': {
                    'rank_in_grp': {'$rank': {}},
                },
            }},
            {'$match': {'rank_in_grp': {'$lte': 3}}},
            {'$project': {'node_id': 1, 'tag': 1, 'grp_id': 1, 'val_int': 1, 'rank_in_grp': 1, '_id': 0}},
            {'$sort': {'grp_id': 1, 'rank_in_grp': 1}},
        ]))

        self._query('window-rank-1', 'dense-rank nodes by val_float within status', lambda s: MongoAggregateQuery('node', [
            {'$setWindowFields': {
                'partitionBy': '$status',
                'sortBy': {'val_float': -1},
                'output': {
                    'dense_rank': {'$denseRank': {}},
                },
            }},
            {'$match': {'dense_rank': {'$lte': 5}}},
            {'$project': {'node_id': 1, 'tag': 1, 'status': 1, 'val_float': 1, 'dense_rank': 1, '_id': 0}},
        ]))

        self._query('window-cumsum-0', 'cumulative log events per node over time', lambda s: MongoAggregateQuery('log', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$sort': {'occurred_at': 1}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'occurred_at': 1},
                'output': {
                    'cumulative_count': {'$sum': 1, 'window': {'documents': ['unbounded', 'current']}},
                },
            }},
            {'$project': {'log_id': 1, 'kind': 1, 'occurred_at': 1, 'cumulative_count': 1, '_id': 0}},
        ]))

        self._query('window-cumsum-1', 'running avg measure val per node+dim', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'node_id': s._param_node_id(), 'dim': s._param_dim()}},
            {'$sort': {'recorded_at': 1}},
            {'$setWindowFields': {
                'partitionBy': {'node_id': '$node_id', 'dim': '$dim'},
                'sortBy': {'recorded_at': 1},
                'output': {
                    'running_avg': {'$avg': '$val', 'window': {'documents': ['unbounded', 'current']}},
                },
            }},
            {'$project': {'measure_id': 1, 'dim': 1, 'val': 1, 'recorded_at': 1, 'running_avg': 1, '_id': 0}},
        ]))

        self._query('window-movavg-0', '3-period moving avg measure val', lambda s: MongoAggregateQuery('measure', [
            {'$match': {'node_id': s._param_node_id(), 'dim': 0}},
            {'$sort': {'recorded_at': 1}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'recorded_at': 1},
                'output': {
                    'moving_avg_3': {'$avg': '$val', 'window': {'documents': [-2, 0]}},
                },
            }},
            {'$project': {'measure_id': 1, 'val': 1, 'recorded_at': 1, 'moving_avg_3': 1, '_id': 0}},
        ]))

        self._query('window-movavg-1', '7-period moving avg log val', lambda s: MongoAggregateQuery('log', [
            {'$match': {'node_id': s._param_node_id(), 'val': {'$ne': None}}},
            {'$sort': {'occurred_at': 1}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'occurred_at': 1},
                'output': {
                    'moving_avg_7': {'$avg': '$val', 'window': {'documents': [-6, 0]}},
                },
            }},
            {'$project': {'log_id': 1, 'val': 1, 'occurred_at': 1, 'moving_avg_7': 1, '_id': 0}},
        ]))

        self._query('window-first-last-0', 'first and last log event per node', lambda s: MongoAggregateQuery('log', [
            {'$match': {'node_id': {'$in': s._param_node_ids(5, 20)}}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'occurred_at': 1},
                'output': {
                    'first_occurred': {'$first': '$occurred_at', 'window': {'documents': ['unbounded', 'unbounded']}},
                    'last_occurred': {'$last': '$occurred_at', 'window': {'documents': ['unbounded', 'unbounded']}},
                },
            }},
            {'$group': {
                '_id': '$node_id',
                'first_occurred': {'$first': '$first_occurred'},
                'last_occurred': {'$first': '$last_occurred'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'count': -1}},
        ]))

        self._query('window-ntile-0', 'quartile of nodes by val_int', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$setWindowFields': {
                'sortBy': {'val_int': 1},
                'output': {
                    'percentile_rank': {'$percentile': {'p': [0.25, 0.5, 0.75], 'method': 'approximate', 'input': '$val_int'}},
                },
            }},
            {'$limit': 1},
        ]))

    #endregion
    #region $facet

    def _register_facet_queries(self):
        self._query('facet-node-0', 'node multi-facet: by status + by grp.depth', lambda s: MongoAggregateQuery('node', [
            {'$match': {'tag': s._param_tag()}},
            {'$facet': {
                'by_status': [
                    {'$group': {'_id': '$status', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                ],
                'by_grp_depth': [
                    {'$group': {'_id': '$grp.depth', 'count': {'$sum': 1}}},
                    {'$sort': {'_id': 1}},
                ],
                'total': [{'$count': 'n'}],
            }},
        ]))

        self._query('facet-node-1', 'node multi-facet: val_int bucket + tag top10', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$facet': {
                'val_int_histogram': [
                    {'$bucket': {
                        'groupBy': '$val_int',
                        'boundaries': [0, 200, 400, 600, 800, 1001],
                        'default': 'Other',
                        'output': {'count': {'$sum': 1}},
                    }},
                ],
                'top_tags': [
                    {'$sortByCount': '$tag'},
                    {'$limit': 10},
                ],
            }},
        ]))

        self._query('facet-log-0', 'log multi-facet: by kind + val histogram', lambda s: MongoAggregateQuery('log', [
            {'$match': {'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$facet': {
                'by_kind': [
                    {'$group': {'_id': '$kind', 'count': {'$sum': 1}, 'avg_val': {'$avg': '$val'}}},
                    {'$sort': {'count': -1}},
                ],
                'val_histogram': [
                    {'$match': {'val': {'$ne': None}}},
                    {'$bucketAuto': {'groupBy': '$val', 'buckets': 5, 'output': {'count': {'$sum': 1}}}},
                ],
                'total': [{'$count': 'n'}],
            }},
        ]))

        self._query('facet-measure-0', 'measure multi-facet: by dim + date histogram', lambda s: MongoAggregateQuery('measure', [
            {'$facet': {
                'by_dim': [
                    {'$group': {'_id': '$dim', 'count': {'$sum': 1}, 'avg_val': {'$avg': '$val'}}},
                    {'$sort': {'_id': 1}},
                ],
                'recent_trend': [
                    {'$match': {'recorded_at': {'$gte': s._param_date_minus_days(30, 90)}}},
                    {'$group': {
                        '_id': {'$dateTrunc': {'date': '$recorded_at', 'unit': 'week'}},
                        'avg_val': {'$avg': '$val'},
                        'count': {'$sum': 1},
                    }},
                    {'$sort': {'_id': 1}},
                ],
            }},
        ]))

        self._query('facet-link-0', 'link facet: by kind + weight distribution', lambda s: MongoAggregateQuery('link', [
            {'$facet': {
                'by_kind': [
                    {'$group': {'_id': '$kind', 'count': {'$sum': 1}, 'avg_weight': {'$avg': '$weight'}}},
                    {'$sort': {'count': -1}},
                ],
                'weight_histogram': [
                    {'$bucketAuto': {'groupBy': '$weight', 'buckets': 5, 'output': {'count': {'$sum': 1}}}},
                ],
            }},
        ]))

    #endregion
    #region Polymorphic

    def _register_event_log_queries(self):
        _EVENT_TYPES = ['status_changed', 'link_added', 'measure_recorded', 'doc_updated', 'tag_changed', 'activated', 'deactivated']
        _ROLES = ['admin', 'user', 'system', 'service']

        # --- PK / point lookup ---
        self._query('event-pk-0', 'event_log by id', lambda s: MongoFindQuery('event_log',
            filter={'event_id': s._param_event_id()},
        ))

        # --- by node ---
        self._query('event-node-0', 'events for a node', lambda s: MongoFindQuery('event_log',
            filter={'node_id': s._param_node_id()},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-node-1', 'events for node + type', lambda s: MongoFindQuery('event_log',
            filter={
                'node_id': s._param_node_id(),
                'event_type': s._param_choice('event_type', _EVENT_TYPES),
            },
            sort={'occurred_at': -1},
        ))

        # --- by event_type ---
        self._query('event-type-0', 'events by type', lambda s: MongoFindQuery('event_log',
            filter={'event_type': s._param_choice('event_type', _EVENT_TYPES)},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-type-1', 'events by type + date', lambda s: MongoFindQuery('event_log',
            filter={
                'event_type': s._param_choice('event_type', ['status_changed', 'activated', 'deactivated']),
                'occurred_at': {'$gte': s._param_date_minus_days(30, 90)},
            },
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        # --- by actor ---
        self._query('event-role-0', 'events by actor.role', lambda s: MongoFindQuery('event_log',
            filter={'actor.role': s._param_choice('role', _ROLES)},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-role-1', 'events by role + type', lambda s: MongoFindQuery('event_log',
            filter={
                'actor.role': s._param_choice('role', ['admin', 'system']),
                'event_type': s._param_choice('event_type', ['status_changed', 'link_added']),
            },
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        # --- date scans ---
        self._query('event-date-0', 'recent events', lambda s: MongoFindQuery('event_log',
            filter={'occurred_at': {'$gte': s._param_date_minus_days(7, 30)}},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-date-1', 'events in date range', lambda s: MongoFindQuery('event_log',
            filter={'occurred_at': {
                '$gte': s._param_date_minus_days(180, 730),
                '$lte': s._param_date_minus_days(0, 90),
            }},
            sort={'occurred_at': 1}
        ))

        # --- polymorphic payload queries ---
        self._query('event-poly-0', 'status_changed events ($exists check)', lambda s: MongoFindQuery('event_log',
            filter={'event_type': 'status_changed', 'payload.from_status': {'$exists': True}},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-poly-1', 'link_added with high weight', lambda s: MongoFindQuery('event_log',
            filter={'event_type': 'link_added', 'payload.weight': {'$gte': 0.7}},
            sort={'payload.weight': -1},
            limit=s._param_limit(),
        ))

        self._query('event-poly-2', 'status_changed with specific from_status', lambda s: MongoFindQuery('event_log',
            filter={'event_type': 'status_changed', 'payload.from_status': s._param_status()},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        self._query('event-poly-3', 'measure_recorded with dim + val range', lambda s: MongoFindQuery('event_log',
            filter={
                'event_type': 'measure_recorded',
                'payload.dim': s._param_dim(),
                'payload.val': {'$gte': 50.0},
            },
            sort={'payload.val': -1},
            limit=s._param_limit(),
        ))

        # --- $type check ---
        self._query('event-type-check-0', '$type check on payload.from_status', lambda s: MongoFindQuery('event_log',
            filter={'payload.from_status': {'$type': 'int'}},
            sort={'occurred_at': -1},
            limit=s._param_limit(),
        ))

        # --- aggregation ---
        self._query('event-agg-0', 'count events by type', lambda s: MongoAggregateQuery('event_log', [
            {'$group': {'_id': '$event_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
        ]))

        self._query('event-agg-1', 'count events by role + type', lambda s: MongoAggregateQuery('event_log', [
            {'$group': {
                '_id': {'role': '$actor.role', 'type': '$event_type'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'count': -1}},
            {'$limit': 20},
        ]))

        self._query('event-timeseries-0', 'events per day', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$group': {
                '_id': {'$dateTrunc': {'date': '$occurred_at', 'unit': 'day'}},
                'count': {'$sum': 1},
                'types': {'$addToSet': '$event_type'}
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('event-timeseries-1', 'events per hour by type', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {
                'event_type': s._param_choice('event_type', ['status_changed', 'link_added']),
                'occurred_at': {'$gte': s._param_date_minus_days(7, 14)},
            }},
            {'$group': {
                '_id': {'$dateTrunc': {'date': '$occurred_at', 'unit': 'hour'}},
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}},
        ]))

        self._query('event-facet-0', 'event log multi-facet', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}}},
            {'$facet': {
                'by_type': [
                    {'$group': {'_id': '$event_type', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                ],
                'by_role': [
                    {'$group': {'_id': '$actor.role', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                ],
                'total': [{'$count': 'n'}],
            }},
        ]))

        self._query('event-lookup-0', 'events with node via $lookup', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {
                'event_type': 'status_changed',
                'occurred_at': {'$gte': s._param_date_minus_days(7, 30)}
            }},
            {'$lookup': {
                'from': 'node', 'localField': 'node_id',
                'foreignField': 'node_id', 'as': 'node',
            }},
            {'$unwind': '$node'},
            {'$project': {
                'event_id': 1, 'event_type': 1, 'occurred_at': 1,
                'payload': 1, 'node.tag': 1, 'node.status': 1, '_id': 0,
            }},
            {'$sort': {'occurred_at': -1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('event-window-0', '$setWindowFields: cumulative events per node', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'node_id': {'$in': s._param_node_ids(5, 15)}}},
            {'$sort': {'node_id': 1, 'occurred_at': 1}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'occurred_at': 1},
                'output': {'cumulative': {
                    '$sum': 1,
                    'window': {'documents': ['unbounded', 'current']},
                }},
            }},
            {'$project': {'event_id': 1, 'node_id': 1, 'event_type': 1, 'occurred_at': 1, 'cumulative': 1, '_id': 0}},
        ]))

    #endregion
    #region Deep nesting

    def _register_node_rich_queries(self):
        _TIERS  = ['bronze', 'silver', 'gold', 'platinum']
        _LABELS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'critical', 'experimental', 'deprecated', 'important', 'temp']

        self._query('rich-pk-0', 'node_rich by id', lambda s: MongoFindQuery('node_rich',
            filter={'node_id': s._param_node_id()},
        ))

        self._query('rich-status-0', 'node_rich by identity.status', lambda s: MongoFindQuery('node_rich',
            filter={'identity.status': s._param_status()},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        self._query('rich-active-0', 'active node_rich sorted by composite score', lambda s: MongoFindQuery('node_rich',
            filter={'identity.is_active': True},
            sort={'scores.composite': -1},
            limit=s._param_limit(),
        ))

        # 3-level flag paths
        self._query('rich-flag-hot-0', 'hot nodes (identity.flags.is_hot, 3-level)', lambda s: MongoFindQuery('node_rich',
            filter={'identity.flags.is_hot': True},
            limit=s._param_limit(),
        ))

        self._query('rich-flag-exp-0', 'active experimental nodes', lambda s: MongoFindQuery('node_rich',
            filter={'identity.flags.is_experimental': True, 'identity.is_active': True},
            limit=s._param_limit(),
        ))

        self._query('rich-flag-dep-0', 'deprecated nodes', lambda s: MongoFindQuery('node_rich',
            filter={'identity.flags.is_deprecated': True},
            projection={'node_id': 1, 'identity.tag': 1, 'identity.status': 1, '_id': 0},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        # Labels array
        self._query('rich-labels-0', 'nodes with specific label in identity.labels', lambda s: MongoFindQuery('node_rich',
            filter={'identity.labels': s._param_choice('label', _LABELS)},
            sort={'scores.composite': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-labels-1', 'nodes with $all of two labels', lambda s: MongoFindQuery('node_rich',
            filter={'identity.labels': {'$all': [
                s._param_choice('label1', ['alpha', 'beta', 'gamma']),
                s._param_choice('label2', ['critical', 'experimental']),
            ]}},
            limit=s._param_limit(),
        ))

        # Scores
        self._query('rich-score-tier-0', 'node_rich by scores.tier', lambda s: MongoFindQuery('node_rich',
            filter={'scores.tier': s._param_choice('tier', _TIERS)},
            sort={'scores.composite': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-score-comp-0', 'node_rich scores.composite >= threshold', lambda s: MongoFindQuery('node_rich',
            filter={'scores.composite': {'$gte': 600.0}},
            sort={'scores.composite': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-score-raw-0', 'scores.raw.val_int range (4-level path)', lambda s: MongoFindQuery('node_rich',
            filter={'scores.raw.val_int': {'$gte': s._param_val_int()}},
            sort={'scores.raw.val_int': -1},
            limit=s._param_limit(),
        ))

        # 4-level context.grp paths
        self._query('rich-grp-depth-0', 'context.grp.depth filter (4-level)', lambda s: MongoFindQuery('node_rich',
            filter={'context.grp.depth': s._param_depth()},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        self._query('rich-grp-priority-0', 'context.grp.priority range (4-level)', lambda s: MongoFindQuery('node_rich',
            filter={'context.grp.priority': {'$gte': s._param_priority()}},
            sort={'context.grp.priority': -1},
            limit=s._param_limit(),
        ))

        # Dimensions dict
        self._query('rich-dim-exists-0', 'nodes with non-null d0 dimension', lambda s: MongoFindQuery('node_rich',
            filter={'dimensions.d0': {'$ne': None}},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        self._query('rich-dim-count-0', 'nodes where d0.count >= threshold (4-level)', lambda s: MongoFindQuery('node_rich',
            filter={'dimensions.d0.count': {'$gte': s._param_int('min_count', 5, 20)}},
            sort={'dimensions.d0.count': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-dim-avg-0', 'nodes where d1.avg >= threshold (4-level)', lambda s: MongoFindQuery('node_rich',
            filter={'dimensions.d1': {'$ne': None}, 'dimensions.d1.avg': {'$gte': 50.0}},
            sort={'dimensions.d1.avg': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-dim-readings-0', 'd0.readings $elemMatch by val (5-level)', lambda s: MongoFindQuery('node_rich',
            filter={'dimensions.d0.readings': {'$elemMatch': {'val': {'$gte': 80.0}}}},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        # Connections
        self._query('rich-conn-outdeg-0', 'hub nodes by connections.out_degree', lambda s: MongoFindQuery('node_rich',
            filter={'connections.out_degree': {'$gte': s._param_int('min_deg', 2, 6)}},
            sort={'connections.out_degree': -1},
            limit=s._param_limit(),
        ))

        self._query('rich-conn-link-0', 'nodes with strong outgoing link of specific kind', lambda s: MongoFindQuery('node_rich',
            filter={'connections.outgoing': {'$elemMatch': {'weight': {'$gte': 0.8}, 'kind': s._param_link_kind()}}},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        # Audit history
        self._query('rich-audit-0', 'nodes with status change in audit.history', lambda s: MongoFindQuery('node_rich',
            filter={'audit.history': {'$elemMatch': {'field': 'status'}}},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        # $objectToArray on dimensions
        self._query('rich-dim-objtoarr-0', '$objectToArray dimensions, list non-null dims', lambda s: MongoAggregateQuery('node_rich', [
            {'$match': {'node_id': {'$in': s._param_node_ids(5, 20)}}},
            {'$project': {
                'node_id': 1,
                'active_dims': {'$filter': {
                    'input': {'$objectToArray': '$dimensions'},
                    'as': 'd',
                    'cond': {'$ne': ['$$d.v', None]},
                }},
            }},
        ]))

        self._query('rich-dim-objtoarr-1', '$objectToArray dimensions, count active dims', lambda s: MongoAggregateQuery('node_rich', [
            {'$match': {'identity.is_active': True, 'node_id': {'$in': s._param_node_ids(20, 50)}}},
            {'$addFields': {'dim_array': {'$filter': {
                'input': {'$objectToArray': '$dimensions'},
                'as': 'd',
                'cond': {'$ne': ['$$d.v', None]},
            }}}},
            {'$addFields': {'active_dim_count': {'$size': '$dim_array'}}},
            {'$sort': {'active_dim_count': -1}},
            {'$project': {'node_id': 1, 'identity.tag': 1, 'active_dim_count': 1, '_id': 0}},
        ]))

        # $replaceRoot
        self._query('rich-replace-root-0', '$replaceRoot: pivot to identity subdoc', lambda s: MongoAggregateQuery('node_rich', [
            {'$match': {'scores.tier': s._param_choice('tier', ['gold', 'platinum'])}},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$identity', {'node_id': '$node_id'}]}}},
            {'$sort': {'status': 1}},
            {'$limit': s._param_limit()},
        ]))

        # $setWindowFields
        self._query('rich-window-0', '$setWindowFields: rank by scores.composite within tier', lambda s: MongoAggregateQuery('node_rich', [
            {'$setWindowFields': {
                'partitionBy': '$scores.tier',
                'sortBy': {'scores.composite': -1},
                'output': {'rank_in_tier': {'$rank': {}}},
            }},
            {'$match': {'rank_in_tier': {'$lte': 5}}},
            {'$project': {'node_id': 1, 'identity.tag': 1, 'scores.tier': 1, 'scores.composite': 1, 'rank_in_tier': 1, '_id': 0}},
            {'$sort': {'scores.tier': 1, 'rank_in_tier': 1}},
        ]))

        # $facet
        self._query('rich-facet-0', 'node_rich multi-facet: tier + out_degree + flags', lambda s: MongoAggregateQuery('node_rich', [
            {'$match': {'identity.is_active': True}},
            {'$facet': {
                'by_tier': [
                    {'$group': {'_id': '$scores.tier', 'count': {'$sum': 1}, 'avg_composite': {'$avg': '$scores.composite'}}},
                    {'$sort': {'avg_composite': -1}},
                ],
                'by_out_degree': [
                    {'$bucketAuto': {'groupBy': '$connections.out_degree', 'buckets': 5, 'output': {'count': {'$sum': 1}}}},
                ],
                'hot_count': [
                    {'$match': {'identity.flags.is_hot': True}},
                    {'$count': 'n'},
                ],
            }},
        ]))

    #endregion
    #region Time-series, Bucket

    def _register_bucket_queries(self):
        self._query('bucket-pk-0', 'bucket by id', lambda s: MongoFindQuery('bucket',
            filter={'bucket_id': s._param_int('bucket_id', 1, s._counts.bucket)},
        ))

        self._query('bucket-node-dim-0', 'buckets for node + dim', lambda s: MongoFindQuery('bucket',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            sort={'hour_start': -1},
        ))

        self._query('bucket-node-0', 'all buckets for a node', lambda s: MongoFindQuery('bucket',
            filter={'node_id': s._param_node_id()},
            sort={'hour_start': -1},
            limit=s._param_limit(),
        ))

        self._query('bucket-dim-0', 'buckets for dim sorted by avg', lambda s: MongoFindQuery('bucket',
            filter={'dim': s._param_dim()},
            sort={'avg': -1},
            limit=s._param_limit(),
        ))

        self._query('bucket-date-0', 'buckets in time range for dim', lambda s: MongoFindQuery('bucket',
            filter={
                'hour_start': {
                    '$gte': s._param_date_minus_days(30, 90),
                    '$lte': s._param_date_minus_days(0, 7)
                },
                'dim': s._param_dim()
            },
            sort={'hour_start': 1},
        ))

        self._query('bucket-stats-avg-0', 'high-avg buckets for dim', lambda s: MongoFindQuery('bucket',
            filter={'avg': {'$gte': s._param_val()}, 'dim': s._param_dim()},
            sort={'avg': -1},
            limit=s._param_limit(),
        ))

        self._query('bucket-stats-stddev-0', 'high-variance buckets', lambda s: MongoFindQuery('bucket',
            filter={'stddev': {'$gte': s._param_float('stddev_thr', 10.0, 30.0)}},
            sort={'stddev': -1},
            limit=s._param_limit(),
        ))

        self._query('bucket-stats-compound-0', 'buckets: node + dim + date + avg', lambda s: MongoFindQuery('bucket',
            filter={
                'node_id': s._param_node_id(),
                'dim': s._param_dim(),
                'hour_start': {'$gte': s._param_date_minus_days(30, 90)},
                'avg': {'$gte': 30.0}
            },
            sort={'hour_start': 1},
        ))

        self._query('bucket-agg-0', 'aggregate bucket stats per node+dim (no $unwind)', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {'node_id': {'$in': s._param_node_ids(10, 30)}}},
            {'$group': {
                '_id': {'node_id': '$node_id', 'dim': '$dim'},
                'total_count': {'$sum': '$count'},
                'overall_avg': {'$avg': '$avg'},
                'overall_min': {'$min': '$min'},
                'overall_max': {'$max': '$max'},
                'n_buckets':   {'$sum': 1},
            }},
            {'$sort': {'overall_avg': -1}},
        ]))

        self._query('bucket-agg-1', 're-bucket: hourly into daily', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {
                'dim': s._param_dim(),
                'hour_start': {'$gte': s._param_date_minus_days(30, 90)},
            }},
            {'$group': {
                '_id': {'node_id': '$node_id', 'day': {'$dateTrunc': {'date': '$hour_start', 'unit': 'day'}}},
                'total_count': {'$sum': '$count'},
                'day_avg':     {'$avg': '$avg'},
                'day_min':     {'$min': '$min'},
                'day_max':     {'$max': '$max'},
                'bucket_count': {'$sum': 1},
            }},
            {'$sort': {'_id.day': 1}},
        ]))

        self._query('bucket-agg-2', 'top nodes by average for dim', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {'dim': s._param_dim()}},
            {'$group': {
                '_id': '$node_id',
                'overall_avg': {'$avg': '$avg'},
                'n_buckets': {'$sum': 1}
            }},
            {'$sort': {'overall_avg': -1}},
            {'$limit': 20},
        ]))

        self._query('bucket-unwind-0', '$unwind bucket data points for node + dim', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {'node_id': s._param_node_id(), 'dim': s._param_dim()}},
            {'$unwind': '$data'},
            {'$project': {
                'node_id': 1,
                'dim': 1,
                'hour_start': 1,
                'offset_s': '$data.offset_s',
                'val': '$data.val',
                'quality': '$data.quality',
                '_id': 0,
            }},
            {'$sort': {'hour_start': 1, 'offset_s': 1}},
        ]))

        self._query('bucket-unwind-1', '$unwind data, filter by quality + group', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {'dim': s._param_dim(), 'stddev': {'$gte': 10.0}}},
            {'$unwind': '$data'},
            {'$match': { 'data.quality': s._param_choice('quality', ['good', 'fair', 'poor', 'suspect'])}},
            {'$group': {
                '_id': {'node_id': '$node_id', 'quality': '$data.quality'},
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$data.val'},
            }},
            {'$sort': {'avg_val': -1}},
        ]))

        self._query('bucket-densify-0', '$densify to fill hourly gaps', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {
                'node_id': s._param_node_id(),
                'dim': 0,
                'hour_start': {'$gte': s._param_date_minus_days(30, 90)},
            }},
            {'$project': {'_id': 0, 'node_id': 1, 'dim': 1, 'hour_start': 1, 'avg': 1}},
            {'$densify': {'field': 'hour_start', 'range': {'step': 1, 'unit': 'hour', 'bounds': 'full'}}},
            {'$sort': {'hour_start': 1}},
        ]))

        self._query('bucket-facet-0', 'bucket multi-facet: by dim + high-variance count', lambda s: MongoAggregateQuery('bucket', [
            {'$facet': {
                'by_dim': [
                    {'$group': {
                        '_id': '$dim',
                        'n_buckets': {'$sum': 1},
                        'avg_avg':   {'$avg': '$avg'},
                        'avg_stddev': {'$avg': '$stddev'},
                    }},
                    {'$sort': {'_id': 1}},
                ],
                'high_variance': [
                    {'$match': {'stddev': {'$gte': 20.0}}},
                    {'$count': 'n'},
                ],
            }},
        ]))

    #endregion
    #region Materialized Path

    def _register_grp_tree_queries(self):

        self._query('grptree-pk-0', 'grp_tree by grp_id', lambda s: MongoFindQuery('grp_tree',
            filter={'grp_id': s._param_grp_id()},
        ))

        self._query('grptree-depth-0', 'root groups (depth 0)', lambda s: MongoFindQuery('grp_tree',
            filter={'depth': 0},
            sort={'stats.node_count': -1},
        ))

        self._query('grptree-depth-1', 'leaf groups (no children)', lambda s: MongoFindQuery('grp_tree',
            filter={'stats.child_count': 0},
            sort={'stats.node_count': -1},
            limit=s._param_limit(),
        ))

        # $regex on path - subtree search
        self._query('grptree-path-0', '$regex on path for subtree search', lambda s: MongoFindQuery('grp_tree',
            filter={'path': {'$regex': f'^/{s._param_int("root_id", 1, s._counts.grp // 5 + 1)}'}},
            sort={'depth': 1, 'priority': -1},
        ))

        self._query('grptree-path-1', '$regex: depth-2 paths only', lambda s: MongoFindQuery('grp_tree',
            filter={'path': {'$regex': r'^\\/\\d+\\/\\d+$'}},
            sort={'priority': -1},
            limit=s._param_limit(),
        ))

        # ancestors array
        self._query('grptree-ancestors-0', 'grp_tree with specific ancestor', lambda s: MongoFindQuery('grp_tree',
            filter={'ancestors': {'$elemMatch': {'grp_id': s._param_grp_id()}}},
            sort={'priority': -1},
        ))

        self._query('grptree-ancestors-1', 'descendants of high-priority ancestor', lambda s: MongoFindQuery('grp_tree',
            filter={'ancestors': {'$elemMatch': {'priority': {'$gte': 0.7}}}, 'depth': {'$gte': 1}},
            sort={'stats.node_count': -1},
            limit=s._param_limit(),
        ))

        # children array
        self._query('grptree-children-0', 'groups with many children', lambda s: MongoFindQuery('grp_tree',
            filter={'stats.child_count': {'$gte': s._param_int('min_children', 3, 10)}},
            sort={'stats.child_count': -1},
            limit=s._param_limit(),
        ))

        self._query('grptree-children-1', 'groups with high-node-count child', lambda s: MongoFindQuery('grp_tree',
            filter={'children': {'$elemMatch': {'node_count': {'$gte': s._param_int('min_nc', 50, 200)}}}},
            sort={'grp_id': 1},
        ))

        # stats
        self._query('grptree-stats-0', 'grp_tree by stats.node_count', lambda s: MongoFindQuery('grp_tree',
            filter={'stats.node_count': {'$gte': s._param_int('min_nc', 30, 100)}},
            sort={'stats.node_count': -1},
        ))

        self._query('grptree-stats-1', 'top groups by active-node ratio', lambda s: MongoAggregateQuery('grp_tree', [
            {'$match': {'stats.node_count': {'$gt': 0}}},
            {'$addFields': {'active_ratio': {'$divide': ['$stats.active_node_count', '$stats.node_count']}}},
            {'$sort': {'active_ratio': -1}},
            {'$limit': 20},
            {'$project': {'grp_id': 1, 'name': 1, 'stats': 1, 'active_ratio': 1, '_id': 0}},
        ]))

        self._query('grptree-unwind-0', '$unwind children, sort by node_count', lambda s: MongoAggregateQuery('grp_tree', [
            {'$match': {'depth': 0, 'stats.child_count': {'$gte': 1}}},
            {'$unwind': '$children'},
            {'$sort': {'children.node_count': -1}},
            {'$project': {
                'grp_id': 1,
                'name': 1,
                'child_grp_id': '$children.grp_id',
                'child_name':   '$children.name',
                'child_nc':     '$children.node_count',
                '_id': 0,
            }},
            {'$limit': s._param_limit()},
        ]))

        self._query('grptree-filter-0', '$filter children by priority threshold', lambda s: MongoAggregateQuery('grp_tree', [
            {'$match': {'depth': 0}},
            {'$project': {
                'grp_id': 1, 'name': 1,
                'high_prio_children': {'$filter': {
                    'input': '$children',
                    'as':    'c',
                    'cond':  {'$gte': ['$$c.priority', 0.7]},
                }},
            }},
            {'$match': {'high_prio_children.0': {'$exists': True}}},
        ]))

        self._query('grptree-path-str-0', 'path length via $strLenCP', lambda s: MongoAggregateQuery('grp_tree', [
            {'$addFields': {'path_len': {'$strLenCP': '$path'}}},
            {'$sort': {'path_len': -1}},
            {'$limit': 20},
            {'$project': {'grp_id': 1, 'name': 1, 'path': 1, 'depth': 1, 'path_len': 1, '_id': 0}},
        ]))

        self._query('grptree-lookup-0', 'leaf groups with their nodes via $lookup', lambda s: MongoAggregateQuery('grp_tree', [
            {'$match': {'stats.child_count': 0, 'stats.node_count': {'$gte': 1}}},
            {'$lookup': {'from': 'node', 'localField': 'grp_id', 'foreignField': 'grp_id', 'as': 'nodes'}},
            {'$addFields': {'actual_node_count': {'$size': '$nodes'}}},
            {'$sort': {'actual_node_count': -1}},
            {'$limit': 10},
            {'$project': {'grp_id': 1, 'name': 1, 'path': 1, 'stats.node_count': 1, 'actual_node_count': 1, '_id': 0}},
        ]))

    #endregion
    #region Cross-cutting

    def _register_advanced_pattern_queries(self):

        # --- $unionWith ---
        self._query('advanced-union-0', '$unionWith: event_log + log for one node', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'node_id': s._param_node_id()}},
            {'$project': {'src': {'$literal': 'event_log'}, 'node_id': 1, 'occurred_at': 1, '_id': 0}},
            {'$unionWith': {'coll': 'log', 'pipeline': [
                {'$match': {'node_id': s._param_node_id()}},
                {'$project': {'src': {'$literal': 'log'}, 'node_id': 1, 'occurred_at': 1, '_id': 0}},
            ]}},
            {'$sort': {'occurred_at': -1}},
            {'$limit': 50},
        ]))

        self._query('advanced-union-1', '$unionWith: merge log + event_log by date', lambda s: MongoAggregateQuery('log', [
            {'$match': {
                'node_id': {'$in': s._param_node_ids(3, 8)},
                'occurred_at': {'$gte': s._param_date_minus_days(30, 90)},
            }},
            {'$project': {'src': {'$literal': 'log'}, 'node_id': 1, 'occurred_at': 1, 'kind': 1, '_id': 0}},
            {'$unionWith': {'coll': 'event_log', 'pipeline': [
                {'$match': {
                    'node_id': {'$in': s._param_node_ids(3, 8)},
                    'occurred_at': {'$gte': s._param_date_minus_days(30, 90)}
                }},
                {'$project': {'src': {'$literal': 'event_log'}, 'node_id': 1, 'occurred_at': 1, 'kind': '$event_type', '_id': 0}},
            ]}},
            {'$sort': {'occurred_at': -1}},
        ]))

        # --- $sample ---
        self._query('advanced-sample-0', '$sample: random node selection', lambda s: MongoAggregateQuery('node', [
            {'$sample': {'size': 100}},
            {'$project': {'node_id': 1, 'tag': 1, 'status': 1, '_id': 0}},
        ]))

        self._query('advanced-sample-1', '$sample: random events of a type', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'event_type': s._param_choice('event_type', ['status_changed', 'link_added'])}},
            {'$sample': {'size': 50}},
            {'$project': {'event_id': 1, 'node_id': 1, 'event_type': 1, 'occurred_at': 1, '_id': 0}},
        ]))

        # --- String operators ---
        self._query('advanced-string-0', '$concat: build label from tag + grp_id', lambda s: MongoAggregateQuery('node', [
            {'$match': {'tag': s._param_tag()}},
            {'$project': {'node_id': 1, 'label': {'$concat': ['$tag', '-', {'$toString': '$grp_id'}]}, '_id': 0}},
            {'$sort': {'node_id': 1}},
            {'$limit': s._param_limit()},
        ]))

        self._query('advanced-string-1', '$toLower on tag, $toString on status', lambda s: MongoAggregateQuery('node', [
            {'$match': {'status': s._param_status()}},
            {'$project': {'node_id': 1, 'tag_lower':  {'$toLower': '$tag'}, 'status_str': {'$toString': '$status'}, '_id': 0}},
            {'$limit': s._param_limit()},
        ]))

        self._query('advanced-string-2', '$regexMatch on node.note', lambda s: MongoAggregateQuery('node', [
            {'$match': {'grp_id': s._param_grp_id()}},
            {'$addFields': {'has_long_word': {'$regexMatch': {'input': '$note', 'regex': r'\w{8,}'}}}},
            {'$match': {'has_long_word': True}},
            {'$project': {'node_id': 1, 'tag': 1, 'note': 1, '_id': 0}},
            {'$limit': s._param_limit()},
        ]))

        self._query('advanced-string-3', '$split on tag field', lambda s: MongoAggregateQuery('node', [
            {'$match': {'is_active': True}},
            {'$project': {'node_id': 1, 'tag': 1, 'tag_parts': {'$split': ['$tag', 'T']}, '_id': 0}},
            {'$limit': s._param_limit()},
        ]))

        # --- Date extraction operators ---
        self._query('advanced-date-0', '$year/$month from log.occurred_at', lambda s: MongoAggregateQuery('log', [
            {'$match': {'kind': s._param_log_kind()}},
            {'$group': {
                '_id': {'year':  {'$year': '$occurred_at'}, 'month': {'$month': '$occurred_at'}},
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$val'},
            }},
            {'$sort': {'_id.year': 1, '_id.month': 1}},
        ]))

        self._query('advanced-date-1', '$dayOfWeek distribution for event_log', lambda s: MongoAggregateQuery('event_log', [
            {'$match': {'event_type': s._param_choice('event_type', ['status_changed', 'link_added'])}},
            {'$group': {'_id': {'$dayOfWeek': '$occurred_at'}, 'count': {'$sum': 1}}},
            {'$sort': {'_id': 1}},
        ]))

        self._query('advanced-date-2', '$hour distribution for bucket.hour_start', lambda s: MongoAggregateQuery('bucket', [
            {'$match': {'dim': s._param_dim()}},
            {'$group': {'_id': {'$hour': '$hour_start'}, 'avg_avg': {'$avg': '$avg'}, 'count': {'$sum': 1}}},
            {'$sort': {'_id': 1}},
        ]))

        self._query('advanced-date-3', '$isoWeek trends for log', lambda s: MongoAggregateQuery('log', [
            {'$match': {'val': {'$ne': None}, 'occurred_at': {'$gte': s._param_date_minus_days(180, 365)}}},
            {'$group': {
                '_id': {'year': {'$isoWeekYear': '$occurred_at'}, 'week': {'$isoWeek': '$occurred_at'}},
                'count': {'$sum': 1},
                'avg_val': {'$avg': '$val'},
            }},
            {'$sort': {'_id.year': 1, '_id.week': 1}},
        ]))

        # --- $sortArray ---
        self._query('advanced-sortarray-0', '$sortArray on out_links by weight', lambda s: MongoAggregateQuery('node', [
            {'$match': {'node_id': {'$in': s._param_node_ids(5, 15)}}},
            {'$project': {
                'node_id': 1,
                'sorted_links': {'$sortArray': {'input': '$out_links', 'sortBy': {'weight': -1}}},
                '_id': 0,
            }},
        ]))

        # --- $mod ---
        self._query('advanced-mod-0', '$mod: val_int divisible by N', lambda s: MongoFindQuery('node',
            filter={'val_int': {'$mod': [s._param_int('divisor', 5, 10), 0]}},
            sort={'val_int': 1},
            limit=s._param_limit(),
        ))

        # --- $regex on node fields ---
        self._query('advanced-regex-0', '$regex on node.note (long-word pattern)', lambda s: MongoFindQuery('node',
            filter={'note': {'$regex': r'\w{8,}', '$options': 'i'}},
            sort={'node_id': 1},
            limit=s._param_limit(),
        ))

        # --- $let variables ---
        self._query('advanced-let-0', '$let variables in $project', lambda s: MongoAggregateQuery('node', [
            {'$match': {'grp_id': s._param_grp_id()}},
            {'$project': {
                'node_id': 1,
                'tag': 1,
                'adjusted_score': {'$let': {
                    'vars': {'base': '$val_int', 'mult': 1.5},
                    'in':   {'$multiply': ['$$base', '$$mult']},
                }},
                '_id': 0,
            }},
            {'$sort': {'adjusted_score': -1}},
            {'$limit': s._param_limit()},
        ]))

        # --- $setUnion / $reduce to merge label sets ---
        self._query('advanced-set-0', '$reduce+$setUnion: all labels per status', lambda s: MongoAggregateQuery('node_rich', [
            {'$match': {'identity.labels': {'$ne': []}}},
            {'$group': {'_id': '$identity.status', 'all_labels': {'$push': '$identity.labels'}}},
            {'$project': {
                '_id': 1,
                'unique_labels': {'$reduce': {
                    'input': '$all_labels',
                    'initialValue': [],
                    'in': {'$setUnion': ['$$value', '$$this']},
                }},
            }},
        ]))

        # --- Window $shift ---
        self._query('advanced-window-shift-0', '$shift: compare log val with previous', lambda s: MongoAggregateQuery('log', [
            {'$match': {'node_id': s._param_node_id(), 'val': {'$ne': None}}},
            {'$sort': {'occurred_at': 1}},
            {'$setWindowFields': {
                'partitionBy': '$node_id',
                'sortBy': {'occurred_at': 1},
                'output': {'prev_val': {'$shift': {'output': '$val', 'by': -1, 'default': None}}},
            }},
            {'$addFields': {'delta': {'$subtract': ['$val', {'$ifNull': ['$prev_val', '$val']}]}}},
            {'$project': {'log_id': 1, 'val': 1, 'occurred_at': 1, 'prev_val': 1, 'delta': 1, '_id': 0}},
        ]))

    #endregion
    #region Write

    def _register_write_queries(self):
        # updateOne / updateMany
        self._write_query('update-0', 'deactivate a node by id', lambda s: MongoUpdateQuery('node',
            filter={'node_id': s._param_node_id()},
            update={'$set': {'is_active': False, 'status': 1}},
            multi=False
        ))

        self._write_query('update-1', 'bulk deactivate nodes by status', lambda s: MongoUpdateQuery('node',
            filter={'status': 4},
            update={'$set': {'is_active': False}},
            multi=True,
        ))

        self._write_query('update-2', 'increment val_int for nodes in grp', lambda s: MongoUpdateQuery('node',
            filter={'grp_id': s._param_grp_id(), 'is_active': True},
            update={'$inc': {'val_int': s._param_int('increment', 1, 10)}},
            multi=True,
        ))

        self._write_query('update-3', 'set tag for a node', lambda s: MongoUpdateQuery('node',
            filter={'node_id': s._param_node_id()},
            update={'$set': {'tag': s._param_tag()}},
            multi=False
        ))

        self._write_query('update-4', 'add log entry for a node via $push', lambda s: MongoUpdateQuery('node',
            filter={'node_id': s._param_node_id()},
            update={'$push': {'logs': {
                'log_id': -1,
                'kind': s._param_log_kind(),
                'val': s._param_val(),
                'occurred_at': '$$NOW',
            }}},
            multi=False
        ))

        self._write_query('update-5', 'remove stale out_links via $pull', lambda s: MongoUpdateQuery('node',
            filter={'node_id': s._param_node_id()},
            update={'$pull': {'out_links': {'kind': s._param_link_kind()}}},
            multi=False
        ))

        self._write_query('update-6', 'bulk update grp priority', lambda s: MongoUpdateQuery('grp',
            filter={'depth': 1, 'priority': {'$lte': 0.1}},
            update={'$set': {'priority': 0.1}},
            multi=True,
        ))

        self._write_query('update-7', 'update measure val for a node', lambda s: MongoUpdateQuery('measure',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            update={'$set': {'val': s._param_val()}},
            multi=True,
        ))

        self._write_query('update-8', 'update link weight', lambda s: MongoUpdateQuery('link',
            filter={'src_id': s._param_node_id(), 'kind': s._param_link_kind()},
            update={'$set': {'weight': s._param_weight()}},
            multi=True,
        ))

        self._write_query('update-9', 'unset val for null-like logs', lambda s: MongoUpdateQuery('log',
            filter={'kind': s._param_log_kind(), 'val': {'$lt': 0.001}},
            update={'$unset': {'val': ''}},
            multi=True,
        ))

        # deleteOne / deleteMany
        self._write_query('delete-0', 'delete a single log entry', lambda s: MongoDeleteQuery('log',
            filter={'log_id': s._param_log_id()},
            multi=False
        ))

        self._write_query('delete-1', 'delete old logs beyond retention window', lambda s: MongoDeleteQuery('log',
            filter={'occurred_at': {'$lt': s._param_date_minus_days(365, 730)}},
            multi=True,
        ))

        self._write_query('delete-2', 'delete measures for a node', lambda s: MongoDeleteQuery('measure',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            multi=True,
        ))

        self._write_query('delete-3', 'delete low-weight links', lambda s: MongoDeleteQuery('link',
            filter={'weight': {'$lt': 0.05}, 'kind': s._param_link_kind()},
            multi=True,
        ))

        # insertOne / insertMany
        self._write_query('insert-0', 'insert a new log entry', lambda s: MongoInsertQuery('log',
            documents=[{
                'log_id': s._param_seed('log'),
                'node_id': s._param_node_id(),
                'kind': s._param_log_kind(),
                'val': s._param_val(),
                'occurred_at': '$$NOW',
            }]
        ))

        self._write_query('insert-1', 'insert a batch of log entries', lambda s: MongoInsertQuery('log',
            documents=[{
                'log_id': s._param_seed('log'),
                'node_id': s._param_node_id(),
                'kind': s._param_log_kind(),
                'val': s._param_val(),
                'occurred_at': '$$NOW',
            } for i in range(20)]
        ))

        self._write_query('insert-2', 'insert a new grp', lambda s: MongoInsertQuery('grp',
            documents=[{
                'grp_id': s._param_seed('grp'),
                'name': 'generated_group',
                'parent_id': s._param_grp_id(),
                'depth': 1,
                'priority': s._param_priority(),
            }]
        ))

        # -- complex updates for node_rich ------------------------------------

        self._write_query('update-10', 'push status-change event into node_rich audit history and update status', lambda s: MongoUpdateQuery('node_rich',
            filter={'node_id': s._param_node_id()},
            update={
                '$set': {'identity.status': s._param_status()},
                '$push': {'audit.history': {
                    'ts': '$$NOW',
                    'field': 'identity.status',
                    'from_val': s._param_status(),
                    'to_val': s._param_status(),
                    'changed_by': 'admin',
                }},
            },
            multi=False,
        ))

        self._write_query('update-11', 'add label to node_rich identity without duplicates via $addToSet', lambda s: MongoUpdateQuery('node_rich',
            filter={'identity.status': s._param_status(), 'context.grp_id': s._param_grp_id()},
            update={'$addToSet': {'identity.labels': s._param_tag()}},
            multi=True,
        ))

        self._write_query('update-12', 'pipeline-update node_rich composite score and tier from raw values', lambda s: MongoUpdateQuery('node_rich',
            filter={'context.grp_id': s._param_grp_id()},
            update={'$set': {
                'scores.composite': {'$add': [
                    '$scores.raw.val_int',
                    {'$multiply': ['$scores.raw.val_float', 100]},
                    {'$multiply': ['$context.grp.priority', 50]},
                ]},
                'scores.tier': {'$switch': {'branches': [
                    {'case': {'$gte': ['$scores.raw.val_int', 900]}, 'then': 'platinum'},
                    {'case': {'$gte': ['$scores.raw.val_int', 700]}, 'then': 'gold'},
                    {'case': {'$gte': ['$scores.raw.val_int', 400]}, 'then': 'silver'},
                ], 'default': 'bronze'}},
                'identity.flags.is_hot': {'$gt': ['$scores.raw.val_int', 800]},
            }},
            multi=True,
        ))

        self._write_query('update-13', 'pull outgoing connections of a given kind from node_rich and decrement degree', lambda s: MongoUpdateQuery('node_rich',
            filter={'node_id': s._param_node_id()},
            update={
                '$pull': {'connections.outgoing': {'kind': s._param_link_kind()}},
                '$inc': {'connections.out_degree': -1},
            },
            multi=False
        ))

        self._write_query('update-14', 'push new outgoing connection into node_rich and increment out_degree', lambda s: MongoUpdateQuery('node_rich',
            filter={'node_id': s._param_node_id()},
            update={
                '$push': {'connections.outgoing': {
                    'dst_id': s._param_node_id('dst_id'),
                    'kind': s._param_link_kind(),
                    'weight': s._param_weight(),
                    'attrs': {'label': 'linked', 'strong': False, 'tags': []},
                }},
                '$inc': {'connections.out_degree': 1},
            },
            multi=False
        ))

        self._write_query('update-15', 'clear a specific dimension sub-document in node_rich', lambda s: MongoUpdateQuery('node_rich',
            filter={'node_id': s._param_node_id()},
            update={'$set': {f'dimensions.d{s._param_dim()}': {
                'count': 0, 'avg': None, 'min': None, 'max': None, 'readings': [],
            }}},
            multi=False
        ))

        # -- complex updates for grp_tree --------------------------------------

        self._write_query('update-16', 'increment node counts in grp_tree stats when a node is added', lambda s: MongoUpdateQuery('grp_tree',
            filter={'grp_id': s._param_grp_id()},
            update={'$inc': {'stats.node_count': 1, 'stats.active_node_count': 1}},
            multi=False
        ))

        self._write_query('update-17', 'push new child entry into grp_tree children array and increment child_count', lambda s: MongoUpdateQuery('grp_tree',
            filter={'grp_id': s._param_grp_id()},
            update={
                '$push': {'children': {
                    'grp_id': s._param_grp_id('child_id'),
                    'name': 'new_subgroup',
                    'priority': s._param_priority(),
                    'node_count': 0,
                }},
                '$inc': {'stats.child_count': 1},
            },
            multi=False
        ))

        self._write_query('update-18', 'pipeline-update to cap priority on all grp_tree descendants of a group', lambda s: MongoUpdateQuery('grp_tree',
            filter={'ancestors.grp_id': s._param_grp_id()},
            update={'$set': {'priority': {'$min': ['$priority', s._param_priority()]}}},
            multi=True,
        ))

        # -- complex updates for event_log -------------------------------------

        self._write_query('update-19', 'mark event_log entries as enriched and stamp processed_at', lambda s: MongoUpdateQuery('event_log',
            filter={'node_id': s._param_node_id(), 'event_type': s._param_choice('event_type', ['measure_recorded', 'doc_updated', 'tag_changed'])},
            update={'$set': {'context.enriched': True, 'context.processed_at': '$$NOW'}},
            multi=True,
        ))

        # -- complex updates for bucket ----------------------------------------

        self._write_query('update-20', 'append a new reading to bucket.data and increment count', lambda s: MongoUpdateQuery('bucket',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            update={
                '$push': {'data': {
                    'offset_s': s._param_int('offset', 0, 3600),
                    'val': s._param_val(),
                    'quality': 'good',
                }},
                '$inc': {'count': 1},
            },
            multi=False
        ))

        self._write_query('update-21', 'pipeline-update to recompute all bucket aggregates from the embedded data array', lambda s: MongoUpdateQuery('bucket',
            filter={'bucket_id': s._param_int('bucket_id', 1, s._counts.bucket)},
            update={'$set': {
                'avg': {'$avg': '$data.val'},
                'min': {'$min': '$data.val'},
                'max': {'$max': '$data.val'},
                'sum': {'$sum': '$data.val'},
                'count': {'$size': '$data'},
            }},
            multi=False
        ))

        self._write_query('update-22', 'pull poor-quality readings from bucket.data', lambda s: MongoUpdateQuery('bucket',
            filter={'node_id': s._param_node_id(), 'dim': s._param_dim()},
            update={'$pull': {'data': {'quality': {'$in': ['bad', 'suspect']}}}},
            multi=True,
        ))

        # -- deletes for new JSON collections ----------------------------------

        self._write_query('delete-4', 'delete event_log entries for a node older than retention window', lambda s: MongoDeleteQuery('event_log',
            filter={'node_id': s._param_node_id(), 'occurred_at': {'$lt': s._param_date_minus_days(180, 365)}},
            multi=True,
        ))

        self._write_query('delete-5', 'delete stale bucket documents outside a rolling time window', lambda s: MongoDeleteQuery('bucket',
            filter={'hour_start': {'$lt': s._param_date_minus_days(30, 90)}, 'dim': s._param_dim()},
            multi=True,
        ))

        self._write_query('delete-6', 'delete low-value event_log entries by event_type and actor role', lambda s: MongoDeleteQuery('event_log',
            filter={
                'event_type': s._param_choice('event_type', ['tag_changed', 'doc_updated']),
                'actor.role': 'guest',
            },
            multi=True,
        ))

        # -- inserts for new JSON collections ----------------------------------

        self._write_query('insert-3', 'insert a new event_log entry with polymorphic status-change payload', lambda s: MongoInsertQuery('event_log',
            documents=[{
                'event_id': s._param_seed('event_log'),
                'node_id': s._param_node_id(),
                'event_type': 'status_changed',
                'occurred_at': '$$NOW',
                'actor': {
                    'user_id': f'u{s._param_int("uid", 1, 999):03d}',
                    'role': s._param_choice('role', ['admin', 'operator', 'viewer']),
                    'session': f'sess_{s._param_int("sess", 1000, 9999)}',
                },
                'payload': {
                    'from_status': s._param_status(),
                    'to_status': s._param_status(),
                    'reason': 'manual',
                },
                'context': {
                    'ip_hash': 'sha256_placeholder',
                    'request_id': f'req_{s._param_int("req", 1, 999999)}',
                    'trace_id': f'trace_{s._param_node_id():06d}',
                },
            }]
        ))

        self._write_query('insert-4', 'insert a bucket document with three embedded hourly readings', lambda s: MongoInsertQuery('bucket',
            documents=[{
                'bucket_id': s._param_seed('bucket'),
                'node_id': s._param_node_id(),
                'dim': s._param_dim(),
                'hour_start': s._param_date_minus_days(1, 30),
                'span_hours': 1,
                'count': 3,
                'min': s._param_val(),
                'max': s._param_val(),
                'sum': s._param_val() * 3,
                'avg': s._param_val(),
                'stddev': s._param_float('stddev', 0.0, 20.0),
                'data': [{'offset_s': i * 1200, 'val': s._param_val(), 'quality': 'good'} for i in range(3)],
            }]
        ))

        self._write_query('insert-5', 'insert a grp_tree node with full ancestor list and zeroed stats', lambda s: MongoInsertQuery('grp_tree',
            documents=[{
                'grp_id': s._param_seed('grp'),
                'name': 'new_subgroup',
                'depth': 2,
                'priority': s._param_priority(),
                'path': f'/{s._param_grp_id()}/{s._param_grp_id("child_id")}',
                'parent_id': s._param_grp_id(),
                'ancestors': [
                    {'grp_id': 1, 'name': 'root', 'depth': 0, 'priority': 1.0},
                    {'grp_id': s._param_grp_id(), 'name': 'parent', 'depth': 1, 'priority': s._param_priority()},
                ],
                'children': [],
                'stats': {'node_count': 0, 'active_node_count': 0, 'child_count': 0},
            }]
        ))

        self._write_query('insert-6', 'insert a full node_rich document with all nested sub-documents', lambda s: MongoInsertQuery('node_rich',
            documents=[{
                'node_id': s._param_seed('node'),
                'identity': {
                    'tag': f'T{s._param_int("tag_num", 1, 9999):04d}',
                    'status': s._param_status(),
                    'is_active': True,
                    'labels': [s._param_tag()],
                    'flags': {'is_hot': False, 'is_experimental': True, 'is_deprecated': False},
                },
                'scores': {
                    'raw': {'val_int': s._param_val_int(), 'val_float': s._param_float('vf', 0.0, 1.0)},
                    'composite': float(s._param_val_int()),
                    'tier': 'bronze',
                },
                'context': {
                    'grp_id': s._param_grp_id(),
                    'grp': {
                        'grp_id': s._param_grp_id(),
                        'name': 'generated',
                        'depth': 1,
                        'priority': s._param_priority(),
                        'parent_id': 1,
                    },
                    'created_at': '$$NOW',
                    'note': None,
                },
                'dimensions': {f'd{d}': None for d in range(5)},
                'connections': {'outgoing': [], 'out_degree': 0, 'in_degree_est': 0},
                'audit': {'schema_version': 2, 'created_by': 'system', 'history': []},
            }]
        ))

    #endregion
