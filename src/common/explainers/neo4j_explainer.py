import sys
import json
import re
from typing import Any
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from common.drivers import Neo4jDriver, cypher

class Neo4jExplainer:
    def __init__(self, driver: Neo4jDriver) -> None:
        self._console = Console()
        self._driver = driver

    def fetch_plan(self, query: str, do_profile: bool) -> dict:
        """
        Fetch query execution plan from Neo4j.
        Args:
            query: Cypher query string
            do_profile: If True, use PROFILE (actually runs the query). If False, use EXPLAIN.
        Returns:
            Plan dictionary
        """
        return fetch_plan(self._driver, query, do_profile)

    def print_tree(self, plan: dict) -> None:
        """Print the plan tree with rich formatting."""
        tree_string = plan_tree_to_string(plan)
        self._console.print(Panel(tree_string, title='[bold]Query Plan Tree[/bold]', border_style='blue'))

    def print_json(self, plan: dict) -> None:
        """Print the plan JSON with syntax highlighting."""
        # Remove internal summary for cleaner JSON output
        copy = {k: v for k, v in plan.items() if not k.startswith('_')}

        json_string = json.dumps(copy, indent=2)
        self._console.print(Panel(
            Syntax(json_string, 'json', theme='monokai', line_numbers=False, word_wrap=False),
            title='[bold]Raw JSON Plan[/bold]',
            border_style='green',
        ))

#region Plan fetching

# Detect DML/write operations in Cypher
DML_RE = re.compile(r'^\s*(CREATE|DELETE|DETACH\s+DELETE|SET|REMOVE|MERGE)\b', re.IGNORECASE | re.MULTILINE)

def fetch_plan(driver: Neo4jDriver, query: str, do_profile: bool) -> dict:
    is_dml = bool(DML_RE.search(query))

    if is_dml:
        if not do_profile:
            print('Note: DML query detected — using EXPLAIN without execution (estimated plan only, query will NOT be executed).\n', file=sys.stderr)
        else:
            print('Note: DML query detected — running inside a transaction that will be rolled back (no data will be modified).\n', file=sys.stderr)

    try:
        with driver.session() as session:
            if do_profile:
                if is_dml:
                    # Use explicit transaction for DML so we can rollback
                    tx = session.begin_transaction()
                    try:
                        result = tx.run(cypher(f'PROFILE {query}'))
                        summary = result.consume()
                        assert summary.profile is not None, 'Failed to retrieve query summary for DML.'
                        plan_dict = normalize_profile(summary.profile)

                        # Add summary statistics
                        plan_dict['_summary'] = {
                            'resultAvailableAfter': summary.result_available_after,
                            'resultConsumedAfter': summary.result_consumed_after,
                        }
                    finally:
                        tx.rollback()  # Always rollback to avoid actual writes
                else:
                    result = session.run(cypher(f'PROFILE {query}'))
                    summary = result.consume()
                    assert summary.profile is not None, 'Failed to retrieve query summary.'
                    plan_dict = normalize_profile(summary.profile)

                    plan_dict['_summary'] = {
                        'resultAvailableAfter': summary.result_available_after,
                        'resultConsumedAfter': summary.result_consumed_after,
                    }
            else:
                # EXPLAIN does not execute the query
                result = session.run(cypher(f'EXPLAIN {query}'))
                summary = result.consume()
                assert summary.plan is not None, 'Failed to retrieve query plan.'
                plan_dict = normalize_plan(summary.plan)

        return plan_dict

    finally:
        driver.close()

def normalize_profile(profile: dict) -> dict[str, Any]:
    """
    Normalize Neo4j profiled plan dictionary to a consistent format.
    Includes actual execution statistics.

    Args:
        profile: Profile dictionary from result summary

    Returns:
        Normalized dictionary representation of the profiled plan
    """
    if not profile:
        return {}

    args = profile.get('args', {})

    result = {
        'operatorType': profile.get('operatorType', 'Unknown'),
        'arguments': args,
        'identifiers': profile.get('identifiers', []),
        'dbHits': profile.get('dbHits', args.get('DbHits')),
        'rows': profile.get('rows', args.get('Rows')),
        'pageCacheHits': profile.get('pageCacheHits', args.get('PageCacheHits')),
        'pageCacheMisses': profile.get('pageCacheMisses', args.get('PageCacheMisses')),
        'time': profile.get('time', args.get('Time')),
    }

    # Handle children recursively
    children = profile.get('children', [])
    if children:
        result['children'] = [normalize_profile(child) for child in children]
    else:
        result['children'] = []

    return result

def normalize_plan(plan: dict) -> dict[str, Any]:
    """
    Normalize Neo4j plan dictionary to a consistent format.
    The plan is already returned as a dict from Neo4j, but we normalize
    field names for consistency (args -> arguments).

    Args:
        plan: Plan dictionary from result summary

    Returns:
        Normalized dictionary representation of the plan
    """
    if not plan:
        return {}

    result = {
        'operatorType': plan.get('operatorType', 'Unknown'),
        'arguments': plan.get('args', {}),
        'identifiers': plan.get('identifiers', []),
    }

    # Handle children recursively
    children = plan.get('children', [])
    if children:
        result['children'] = [normalize_plan(child) for child in children]
    else:
        result['children'] = []

    return result

#endregion
#region Printing

def plan_tree_to_string(plan: dict) -> str:
    """Return the full visual tree as a single string."""
    header_parts = []

    # Summary statistics from PROFILE
    summary = plan.get('_summary', {})
    if 'resultAvailableAfter' in summary:
        header_parts.append(f'Planning: {summary["resultAvailableAfter"]} ms')
    if 'resultConsumedAfter' in summary:
        header_parts.append(f'Execution: {summary["resultConsumedAfter"]} ms')

    header = ''
    if header_parts:
        header = '  ' + ' | '.join(header_parts) + '\n\n'

    lines = _render_tree(plan)
    return header + '\n'.join(lines)

def _render_tree(node: dict, prefix: str = '', is_last: bool = True) -> list[str]:
    """Recursively render the plan tree into a list of lines."""
    connector = "└─ " if is_last else "├─ "
    lines = [prefix + connector + _node_label(node)]

    child_prefix = prefix + ("   " if is_last else "│  ")

    children: list[dict] = node.get('children', [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        lines.extend(_render_tree(child, child_prefix, last))

    return lines

def _node_label(node: dict) -> str:
    """Generate a descriptive label for a plan node."""
    op_type = _clean_operator_type(node.get('operatorType', '?'))
    icon = NODE_ICONS.get(op_type, _abbreviate_operator(op_type))
    label = icon

    extras: list[str] = []
    args = node.get('arguments', {})

    # Label for scans
    if 'LabelName' in args:
        extras.append(f':{args["LabelName"]}')

    # Index name
    if 'Index' in args:
        extras.append(f'idx={args["Index"]}')

    # Details (filter, predicate, etc.)
    if 'Details' in args:
        details = str(args['Details'])
        if len(details) > 70:
            details = details[:67] + '...'
        extras.append(details)

    # Expression
    if 'Expression' in args:
        expr = str(args['Expression'])
        if len(expr) > 60:
            expr = expr[:57] + '...'
        extras.append(expr)

    # Order
    if 'Order' in args:
        order = str(args['Order'])
        if len(order) > 40:
            order = order[:37] + '...'
        extras.append(f'by={order}')

    # Relationship types
    if 'RelationshipTypes' in args:
        rel_types = args['RelationshipTypes']
        if rel_types:
            extras.append(f'rels={rel_types}')

    # Identifiers are shown only if extras is empty (to keep label readable)
    identifiers = node.get('identifiers', [])
    if identifiers and not extras:
        id_str = ', '.join(identifiers[:3])
        if len(identifiers) > 3:
            id_str += '...'
        extras.append(f'({id_str})')

    if extras:
        label += ' ' + ' '.join(extras)

    label += _fmt_stats(node)
    return label

NODE_ICONS: dict[str, str] = {
    # Scan operations
    'AllNodesScan': 'ALLSCAN',
    'NodeByLabelScan': 'LBLSCAN',
    'NodeIndexScan': 'IDXSCAN',
    'NodeIndexSeek': 'IDXSEEK',
    'NodeUniqueIndexSeek': 'UIDXSEEK',
    'DirectedRelationshipIndexScan': 'RELIDXSCAN',
    'DirectedRelationshipIndexSeek': 'RELIDXSEEK',
    'NodeByIdSeek': 'IDSEEK',
    'DirectedRelationshipByIdSeek': 'RELIDSEEK',
    'UndirectedRelationshipByIdSeek': 'URELIDSEEK',
    'NodeIndexContainsScan': 'CONTAINSCAN',
    'NodeIndexEndsWithScan': 'ENDSCAN',

    # Count store operations
    'NodeCountFromCountStore': 'CNTSTORE',
    'RelationshipCountFromCountStore': 'RELCNTSTORE',

    # Expand operations
    'Expand(All)': 'EXPAND',
    'Expand(Into)': 'EXPINTO',
    'OptionalExpand(All)': 'OPTEXP',
    'OptionalExpand(Into)': 'OPTEXPINTO',
    'VarLengthExpand(All)': 'VAREXP',
    'VarLengthExpand(Into)': 'VAREXPINTO',
    'VarLengthExpand(Pruning)': 'PRUNEXP',

    # Filter and predicates
    'Filter': 'FILTER',
    'Argument': 'ARG',
    'Selection': 'SELECT',

    # Aggregation and grouping
    'EagerAggregation': 'EAGERAGG',
    'OrderedAggregation': 'ORDAGG',
    'Aggregation': 'AGG',
    'Distinct': 'DISTINCT',

    # Sorting and limiting
    'Sort': 'SORT',
    'Top': 'TOP',
    'Skip': 'SKIP',
    'Limit': 'LIMIT',
    'PartialSort': 'PARTSORT',

    # Join operations
    'NodeHashJoin': 'HJOIN',
    'ValueHashJoin': 'VHJOIN',
    'NodeLeftOuterHashJoin': 'LOJOIN',
    'NodeRightOuterHashJoin': 'ROJOIN',
    'CartesianProduct': 'CROSS',
    'Apply': 'APPLY',
    'SemiApply': 'SEMIAPPLY',
    'AntiSemiApply': 'ANTISEMI',
    'SelectOrSemiApply': 'SELORSA',
    'SelectOrAntiSemiApply': 'SELORAS',
    'LetSemiApply': 'LETSA',
    'LetAntiSemiApply': 'LETASA',
    'ConditionalApply': 'CONDAPPLY',
    'RollUpApply': 'ROLLUP',
    'ForeachApply': 'FOREACH',

    # Set operations
    'Union': 'UNION',
    'OrderedUnion': 'ORDUNION',

    # Results and projection
    'ProduceResults': 'RESULTS',
    'Projection': 'PROJECT',
    'Eager': 'EAGER',
    'CacheProperties': 'CACHE',

    # Write operations
    'Create': 'CREATE',
    'Merge': 'MERGE',
    'Delete': 'DELETE',
    'DetachDelete': 'DETDELETE',
    'SetProperty': 'SETPROP',
    'SetNodeProperty': 'SETNPROP',
    'SetRelationshipProperty': 'SETRPROP',
    'SetLabels': 'SETLABELS',
    'RemoveLabels': 'RMLABELS',
    'SetNodePropertiesFromMap': 'SETMAP',
    'SetRelationshipPropertiesFromMap': 'SETRMAP',
    'CreateNode': 'CRNODE',
    'CreateRelationship': 'CRREL',
    'EmptyResult': 'EMPTY',

    # Locking
    'LockNodes': 'LOCK',

    # Procedures and functions
    'ProcedureCall': 'PROC',

    # Subqueries
    'SubqueryForeach': 'SUBQFE',
    'TransactionForeach': 'TXFE',
    'TransactionApply': 'TXAPPLY',

    # Other
    'LoadCSV': 'LOADCSV',
    'Unwind': 'UNWIND',
    'Optional': 'OPTIONAL',
    'AntiConditionalApply': 'ANTICOND',
    'AssertSameNode': 'ASSERTN',
    'TriadicSelection': 'TRIADIC',
    'TriadicBuild': 'TRIBUILD',
    'TriadicFilter': 'TRIFILT',
    'Input': 'INPUT',
}

def _clean_operator_type(op_type: str) -> str:
    """Remove @neo4j suffix from operator type."""
    # Remove common suffixes like @neo4j
    if '@' in op_type:
        op_type = op_type.split('@')[0]
    return op_type

def _abbreviate_operator(op_type: str) -> str:
    """Create a short abbreviation for operator types not in NODE_ICONS."""
    # Split CamelCase into words and take first letters
    words = re.findall(r'[A-Z][a-z]*', op_type)
    if words:
        # Take first 2 chars of each word, join them
        abbrev = ''.join(w[:2].upper() for w in words[:4])
        return abbrev if len(abbrev) <= 10 else abbrev[:10]
    return op_type[:10].upper()

def _fmt_stats(node: dict) -> str:
    """Format execution statistics for a node."""
    parts = []

    # Estimated rows (from EXPLAIN)
    args = node.get('arguments', {})
    if 'EstimatedRows' in args:
        est = args['EstimatedRows']
        if isinstance(est, float):
            parts.append(f'est={est:.1f}')
        else:
            parts.append(f'est={est}')

    # Actual rows and db hits (from PROFILE)
    if 'rows' in node and node['rows'] is not None:
        parts.append(f'rows={node["rows"]}')
    if 'dbHits' in node and node['dbHits'] is not None:
        parts.append(f'hits={node["dbHits"]}')

    # Page cache (from PROFILE, if available)
    pc_hits = node.get('pageCacheHits')
    pc_misses = node.get('pageCacheMisses')
    if pc_hits is not None and pc_misses is not None:
        total = pc_hits + pc_misses
        if total > 0:
            ratio = pc_hits / total * 100
            parts.append(f'cache={ratio:.0f}%')

    return '  ' + f'[{", ".join(parts)}]' if parts else ''

#endregion
