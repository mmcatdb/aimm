import sys
import json
import re
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from common.drivers import PostgresDriver

class PostgresExplainer:
    def __init__(self, driver: PostgresDriver) -> None:
        self._console = Console()
        self._driver = driver

    def fetch_plan(self, query: str, do_profile: bool, do_discard: bool) -> dict:
        return fetch_plan(self._driver, query, do_profile, do_discard)

    def print_tree(self, plan: dict) -> None:
        """Print the plan tree with rich formatting."""
        tree_string = plan_tree_to_string(plan)
        self._console.print(Panel(tree_string, title='[bold]Query Plan Tree[/bold]', border_style='blue'))

    def print_json(self, plan: dict) -> None:
        """Print the plan JSON with syntax highlighting."""
        json_string = json.dumps(plan, indent=2)
        self._console.print(Panel(
            Syntax(json_string, 'json', theme='monokai', line_numbers=False, word_wrap=False),
            title='[bold]Raw JSON Plan[/bold]',
            border_style='green',
        ))

#region Plan fetching

# Detect DML/write operations in SQL
DML_RE = re.compile(r'^\s*(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|CREATE|DROP|ALTER)\b', re.IGNORECASE)

def fetch_plan(driver: PostgresDriver, query: str, do_profile: bool, do_discard: bool) -> dict:
    """Return the root plan dict from PostgreSQL EXPLAIN … FORMAT JSON."""
    is_dml = bool(DML_RE.match(query))

    if is_dml:
        if do_profile:
            print('Note: DML query detected — running inside a transaction that will be rolled back (no data will be modified).\n', file=sys.stderr)
        else:
            print('Note: DML query detected — using EXPLAIN without ANALYZE (estimated plan only, query will NOT be executed).\n', file=sys.stderr)

    # For DML with ANALYZE we wrap in a transaction and roll back so nothing is actually committed to the database.
    is_in_transaction = is_dml and do_profile

    connection = driver.get_connection()

    try:
        # Rollback needs explicit transaction.
        connection.autocommit = not is_in_transaction

        with connection.cursor() as cursor:
            if is_in_transaction:
                cursor.execute('BEGIN;')

            if do_discard and do_profile:
                try:
                    cursor.execute('DISCARD ALL;')
                except Exception as e:
                    print(f'Warning: DISCARD ALL failed: {e}', file=sys.stderr)

            options = 'FORMAT JSON, VERBOSE, COSTS'
            if do_profile:
                options += ', ANALYZE, BUFFERS'

            cursor.execute(f'EXPLAIN ({options}) {query}')
            row = cursor.fetchone()

            if is_in_transaction:
                cursor.execute('ROLLBACK;')
                if not connection.autocommit:
                    connection.rollback()

            assert row is not None, 'No plan returned from EXPLAIN.'

        # psycopg2 returns the JSON array; index 0 is the plan root
        plan_json = row[0]
        if isinstance(plan_json, list):
            plan_json = plan_json[0]

        return plan_json

    finally:
        driver.put_connection(connection)

#endregion
#region Printing

def plan_tree_to_string(plan: dict) -> str:
    """Return the full visual tree as a single string."""
    root_node = plan.get('Plan', plan)  # handle both wrapped and bare plans

    header_parts = []
    if 'Planning Time' in plan:
        header_parts.append(f'Planning: {plan["Planning Time"]:.3f} ms')
    if 'Execution Time' in plan:
        header_parts.append(f'Execution: {plan["Execution Time"]:.3f} ms')

    header = ''
    if header_parts:
        header = '  ' + ' | '.join(header_parts) + '\n\n'

    lines = _render_tree(root_node)
    return header + '\n'.join(lines)

def _render_tree(node: dict, prefix: str = '', is_last: bool = True) -> list[str]:
    """Recursively render the plan tree into a list of lines."""
    connector = '└─ ' if is_last else '├─ '
    lines = [prefix + connector + _node_label(node)]

    child_prefix = prefix + ('   ' if is_last else '│  ')

    children: list[dict] = node.get('Plans', [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        lines.extend(_render_tree(child, child_prefix, last))

    return lines

def _node_label(node: dict) -> str:
    kind = node.get('Node Type', '?')
    icon = NODE_ICONS.get(kind, kind.upper()[:8])
    assert icon is not None, f'Unknown node type: {kind}'
    label = icon

    extras: list[str] = []
    for key in ('Relation Name', 'Index Name', 'CTE Name', 'Function Name', 'Schema', 'Operation'):
        if key in node:
            extras.append(str(node[key]))
    if 'Filter' in node:
        filt = node['Filter']
        if len(filt) > 60:
            filt = filt[:57] + '...'
        extras.append(f'filter={filt}')
    if 'Join Filter' in node:
        jf = node['Join Filter']
        if len(jf) > 60:
            jf = jf[:57] + '...'
        extras.append(f'join={jf}')
    if 'Hash Cond' in node:
        hc = node['Hash Cond']
        if len(hc) > 60:
            hc = hc[:57] + '...'
        extras.append(f'on={hc}')
    if 'Sort Key' in node:
        sk = ', '.join(node['Sort Key'])
        extras.append(f'by={sk[:50]}')
    if 'Group Key' in node:
        gk = ', '.join(node['Group Key'])
        extras.append(f'by={gk[:50]}')

    if extras:
        label += ' ' + ' '.join(extras)
    label += _fmt_cost(node)
    return label

# Node-type -> short label
NODE_ICONS: dict[str, str] = {
    'Seq Scan':              'SCAN',
    'Index Scan':            'ISCAN',
    'Index Only Scan':       'IOSCAN',
    'Bitmap Heap Scan':      'BHSCAN',
    'Bitmap Index Scan':     'BISCAN',
    'Hash Join':             'HJOIN',
    'Merge Join':            'MJOIN',
    'Nested Loop':           'NLOOP',
    'Hash':                  'HASH',
    'Sort':                  'SORT',
    'Aggregate':             'AGG',
    'Group':                 'GROUP',
    'Limit':                 'LIMIT',
    'Subquery Scan':         'SUBSCAN',
    'CTE Scan':              'CTESCAN',
    'Materialize':           'MAT',
    'Memoize':               'MEMO',
    'Result':                'RESULT',
    'Append':                'APPEND',
    'Merge Append':          'MAPPEND',
    'Gather':                'GATHER',
    'Gather Merge':          'GMERGE',
    'Incremental Sort':      'ISORT',
    'Unique':                'UNIQ',
    'SetOp':                 'SETOP',
    'LockRows':              'LOCK',
    'ModifyTable':           'MODIFY',
    'Insert':                'INSERT',
    'Update':                'UPDATE',
    'Delete':                'DELETE',
    'Merge':                 'MERGE',
    'WindowAgg':             'WINAGG',
    'Values Scan':           'VALUES',
    'Function Scan':         'FUNCSCAN',
    'TableFunc Scan':        'TFSCAN',
    'WorkTable Scan':        'WTSCAN',
    'Foreign Scan':          'FSCAN',
    'Custom Scan':           'CSCAN',
    'BitmapAnd':             'BAND',
    'BitmapOr':              'BOR',
    'ProjectSet':            'PROJSET',
    'Recursive Union':       'RECUNION',
}

def _fmt_cost(node: dict) -> str:
    """Return a short cost/time annotation for a node."""
    parts = []
    if 'Total Cost' in node:
        parts.append(f'cost={node.get("Startup Cost", 0):.1f}..{node["Total Cost"]:.1f}')
    if 'Actual Total Time' in node:
        rows = node.get('Actual Rows', '?')
        loops = node.get('Actual Loops', 1)
        t = node['Actual Total Time']
        parts.append(f'time={t:.3f}ms rows={rows}×{loops}')
    elif 'Plan Rows' in node:
        parts.append(f'rows≈{node["Plan Rows"]}')
    return '  ' + f'[{", ".join(parts)}]' if parts else ''

#endregion
