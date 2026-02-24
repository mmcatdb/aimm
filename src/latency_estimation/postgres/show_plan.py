"""
Example usage:
  python show_plan.py --tree-only "UPDATE orders SET o_totalprice = 0 WHERE o_orderkey = 1"
"""


import sys
import json
import textwrap
import re
import argparse
import psycopg2
import yaml
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()


CONFIG_PATH = "config.yaml"

def load_config(path: str = CONFIG_PATH) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)["postgres"]


def get_connection(cfg: Dict):
    return psycopg2.connect(
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
    )


# ── Plan fetching ─────────────────────────────────────────────────────────────

DML_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|CREATE|DROP|ALTER)\b",
    re.IGNORECASE,
)


def fetch_plan(query: str, analyze: bool, discard: bool) -> Dict:
    """Return the root plan dict from PostgreSQL EXPLAIN … FORMAT JSON."""
    cfg = load_config()
    conn = get_connection(cfg)

    is_dml = bool(DML_RE.match(query))
    # For DML with ANALYZE we wrap in a transaction and roll back so nothing
    # is actually committed to the database.
    wrap_in_tx = analyze and is_dml

    try:
        conn.autocommit = not wrap_in_tx          # need explicit tx for rollback

        with conn.cursor() as cur:
            if wrap_in_tx:
                cur.execute("BEGIN;")

            if discard and analyze:
                try:
                    cur.execute("DISCARD ALL;")
                except Exception as e:
                    print(f"Warning: DISCARD ALL failed: {e}", file=sys.stderr)

            options = "FORMAT JSON, VERBOSE, COSTS"
            if analyze:
                options += ", ANALYZE, BUFFERS"

            cur.execute(f"EXPLAIN ({options}) {query}")
            row = cur.fetchone()

            if wrap_in_tx:
                cur.execute("ROLLBACK;")
                if not conn.autocommit:
                    conn.rollback()

        # psycopg2 returns the JSON array; index 0 is the plan root
        plan_json = row[0]
        if isinstance(plan_json, list):
            plan_json = plan_json[0]
        return plan_json

    finally:
        conn.close()


# Node-type → short label
NODE_ICONS: Dict[str, str] = {
    "Seq Scan":              "SCAN",
    "Index Scan":            "ISCAN",
    "Index Only Scan":       "IOSCAN",
    "Bitmap Heap Scan":      "BHSCAN",
    "Bitmap Index Scan":     "BISCAN",
    "Hash Join":             "HJOIN",
    "Merge Join":            "MJOIN",
    "Nested Loop":           "NLOOP",
    "Hash":                  "HASH",
    "Sort":                  "SORT",
    "Aggregate":             "AGG",
    "Group":                 "GROUP",
    "Limit":                 "LIMIT",
    "Subquery Scan":         "SUBSCAN",
    "CTE Scan":              "CTESCAN",
    "Materialize":           "MAT",
    "Memoize":               "MEMO",
    "Result":                "RESULT",
    "Append":                "APPEND",
    "Merge Append":          "MAPPEND",
    "Gather":                "GATHER",
    "Gather Merge":          "GMERGE",
    "Incremental Sort":      "ISORT",
    "Unique":                "UNIQ",
    "SetOp":                 "SETOP",
    "LockRows":              "LOCK",
    "ModifyTable":           "MODIFY",
    "Insert":                "INSERT",
    "Update":                "UPDATE",
    "Delete":                "DELETE",
    "Merge":                 "MERGE",
    "WindowAgg":             "WINAGG",
    "Values Scan":           "VALUES",
    "Function Scan":         "FUNCSCAN",
    "TableFunc Scan":        "TFSCAN",
    "WorkTable Scan":        "WTSCAN",
    "Foreign Scan":          "FSCAN",
    "Custom Scan":           "CSCAN",
    "BitmapAnd":             "BAND",
    "BitmapOr":              "BOR",
    "ProjectSet":            "PROJSET",
    "Recursive Union":       "RECUNION",
}


def _fmt_cost(node: Dict) -> str:
    """Return a short cost/time annotation for a node."""
    parts = []
    if "Total Cost" in node:
        parts.append(f"cost={node.get('Startup Cost', 0):.1f}..{node['Total Cost']:.1f}")
    if "Actual Total Time" in node:
        rows = node.get("Actual Rows", "?")
        loops = node.get("Actual Loops", 1)
        t = node["Actual Total Time"]
        parts.append(f"time={t:.3f}ms rows={rows}×{loops}")
    elif "Plan Rows" in node:
        parts.append(f"rows≈{node['Plan Rows']}")
    return "  " + f"[{', '.join(parts)}]" if parts else ""


def _node_label(node: Dict) -> str:
    kind = node.get("Node Type", "?")
    icon = NODE_ICONS.get(kind, kind.upper()[:8])
    label = icon

    extras: List[str] = []
    for key in ("Relation Name", "Index Name", "CTE Name",
                 "Function Name", "Schema", "Operation"):
        if key in node:
            extras.append(str(node[key]))
    if "Filter" in node:
        filt = node["Filter"]
        if len(filt) > 60:
            filt = filt[:57] + "..."
        extras.append(f"filter={filt}")
    if "Join Filter" in node:
        jf = node["Join Filter"]
        if len(jf) > 60:
            jf = jf[:57] + "..."
        extras.append(f"join={jf}")
    if "Hash Cond" in node:
        hc = node["Hash Cond"]
        if len(hc) > 60:
            hc = hc[:57] + "..."
        extras.append(f"on={hc}")
    if "Sort Key" in node:
        sk = ", ".join(node["Sort Key"])
        extras.append(f"by={sk[:50]}")
    if "Group Key" in node:
        gk = ", ".join(node["Group Key"])
        extras.append(f"by={gk[:50]}")

    if extras:
        label += " " + " ".join(extras)
    label += _fmt_cost(node)
    return label


def _render_tree(node: Dict, prefix: str = "", is_last: bool = True) -> List[str]:
    """Recursively render the plan tree into a list of lines."""
    connector = "└─ " if is_last else "├─ "
    lines = [prefix + connector + _node_label(node)]

    child_prefix = prefix + ("   " if is_last else "│  ")

    children: List[Dict] = node.get("Plans", [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        lines.extend(_render_tree(child, child_prefix, last))

    return lines


def render_plan_tree(plan: Dict) -> str:
    """Return the full visual tree as a single string."""
    root_node = plan.get("Plan", plan)  # handle both wrapped and bare plans

    header_parts = []
    if "Planning Time" in plan:
        header_parts.append(f"Planning: {plan['Planning Time']:.3f} ms")
    if "Execution Time" in plan:
        header_parts.append(f"Execution: {plan['Execution Time']:.3f} ms")

    header = ""
    if header_parts:
        header = "  " + " | ".join(header_parts) + "\n\n"

    lines = _render_tree(root_node)
    return header + "\n".join(lines)



def print_tree(plan: Dict) -> None:
    tree_str = render_plan_tree(plan)
    console.print(Panel(tree_str, title="[bold]Query Plan Tree[/bold]",
                        border_style="blue"))


def print_json(plan: Dict) -> None:
    json_str = json.dumps(plan, indent=2)
    console.print(Panel(
        Syntax(json_str, "json", theme="monokai", line_numbers=False,
               word_wrap=False),
        title="[bold]Raw JSON Plan[/bold]",
        border_style="green",
    ))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a PostgreSQL query plan visually.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="SQL query string. Read from stdin if omitted.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Use EXPLAIN ANALYZE (actually runs the query; DML is rolled back).",
    )
    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Issue DISCARD ALL before running (requires --analyze).",
    )
    parser.add_argument(
        "--json-only",
        dest="json_only",
        action="store_true",
        help="Print only the raw JSON, skip the visual tree.",
    )
    parser.add_argument(
        "--tree-only",
        dest="tree_only",
        action="store_true",
        help="Print only the visual tree, skip the raw JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        print("Enter your SQL query (finish with Ctrl-D on a blank line):")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        query = "\n".join(lines).strip()

    if not query:
        print("Error: no query provided.", file=sys.stderr)
        sys.exit(1)

    if DML_RE.match(query):
        if args.analyze:
            print(
                "Note: DML query detected — running inside a transaction "
                "that will be rolled back (no data will be modified).\n",
                file=sys.stderr,
            )
        else:
            print(
                "Note: DML query detected — using EXPLAIN without ANALYZE "
                "(estimated plan only, query will NOT be executed).\n",
                file=sys.stderr,
            )

    try:
        plan = fetch_plan(query, analyze=args.analyze, discard=args.no_cache)
    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    if not args.json_only:
        print_tree(plan)

    if not args.tree_only:
        print()
        print_json(plan)


if __name__ == "__main__":
    main()
