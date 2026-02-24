"""
Example usage:
    python show_plan.py "MATCH (c:Customer)-[:PLACED]->(o:Order) WHERE o.o_totalprice > 100000 SET c.vip = true"
"""

import sys
import json
import textwrap
import re
import argparse
import yaml
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()

CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH) -> Dict:
    """Load Neo4j configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)["neo4j"]


def get_driver(cfg: Dict):
    """Create Neo4j driver connection."""
    return GraphDatabase.driver(
        cfg["uri"],
        auth=(cfg["user"], cfg["password"])
    )



# Detect DML/write operations in Cypher
DML_RE = re.compile(
    r"^\s*(CREATE|DELETE|DETACH\s+DELETE|SET|REMOVE|MERGE)\b",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_plan(plan: Dict) -> Dict[str, Any]:
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
        "operatorType": plan.get("operatorType", "Unknown"),
        "arguments": plan.get("args", {}),
        "identifiers": plan.get("identifiers", []),
    }
    
    # Handle children recursively
    children = plan.get("children", [])
    if children:
        result["children"] = [normalize_plan(child) for child in children]
    else:
        result["children"] = []
    
    return result


def normalize_profile(profile: Dict) -> Dict[str, Any]:
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
    
    args = profile.get("args", {})
    
    result = {
        "operatorType": profile.get("operatorType", "Unknown"),
        "arguments": args,
        "identifiers": profile.get("identifiers", []),
        "dbHits": profile.get("dbHits", args.get("DbHits")),
        "rows": profile.get("rows", args.get("Rows")),
        "pageCacheHits": profile.get("pageCacheHits", args.get("PageCacheHits")),
        "pageCacheMisses": profile.get("pageCacheMisses", args.get("PageCacheMisses")),
        "time": profile.get("time", args.get("Time")),
    }
    
    # Handle children recursively
    children = profile.get("children", [])
    if children:
        result["children"] = [normalize_profile(child) for child in children]
    else:
        result["children"] = []
    
    return result


def fetch_plan(query: str, profile: bool = False, rollback_dml: bool = True) -> Dict:
    """
    Fetch query execution plan from Neo4j.
    
    Args:
        query: Cypher query string
        profile: If True, use PROFILE (actually runs the query). If False, use EXPLAIN.
        rollback_dml: If True, rollback DML operations after profiling
        
    Returns:
        Plan dictionary
    """
    cfg = load_config()
    driver = get_driver(cfg)
    
    is_dml = bool(DML_RE.search(query))
    
    try:
        with driver.session() as session:
            if profile:
                if is_dml and rollback_dml:
                    # Use explicit transaction for DML so we can rollback
                    tx = session.begin_transaction()
                    try:
                        result = tx.run(f"PROFILE {query}")
                        summary = result.consume()
                        plan_dict = normalize_profile(summary.profile)
                        
                        # Add summary statistics
                        plan_dict["_summary"] = {
                            "resultAvailableAfter": summary.result_available_after,
                            "resultConsumedAfter": summary.result_consumed_after,
                        }
                    finally:
                        tx.rollback()  # Always rollback to avoid actual writes
                else:
                    result = session.run(f"PROFILE {query}")
                    summary = result.consume()
                    plan_dict = normalize_profile(summary.profile)
                    
                    plan_dict["_summary"] = {
                        "resultAvailableAfter": summary.result_available_after,
                        "resultConsumedAfter": summary.result_consumed_after,
                    }
            else:
                # EXPLAIN does not execute the query
                result = session.run(f"EXPLAIN {query}")
                summary = result.consume()
                plan_dict = normalize_plan(summary.plan)
        
        return plan_dict
    
    finally:
        driver.close()


# ── Node type icons ───────────────────────────────────────────────────────────

NODE_ICONS: Dict[str, str] = {
    # Scan operations
    "AllNodesScan": "ALLSCAN",
    "NodeByLabelScan": "LBLSCAN",
    "NodeIndexScan": "IDXSCAN",
    "NodeIndexSeek": "IDXSEEK",
    "NodeUniqueIndexSeek": "UIDXSEEK",
    "DirectedRelationshipIndexScan": "RELIDXSCAN",
    "DirectedRelationshipIndexSeek": "RELIDXSEEK",
    "NodeByIdSeek": "IDSEEK",
    "DirectedRelationshipByIdSeek": "RELIDSEEK",
    "UndirectedRelationshipByIdSeek": "URELIDSEEK",
    "NodeIndexContainsScan": "CONTAINSCAN",
    "NodeIndexEndsWithScan": "ENDSCAN",
    
    # Count store operations
    "NodeCountFromCountStore": "CNTSTORE",
    "RelationshipCountFromCountStore": "RELCNTSTORE",
    
    # Expand operations
    "Expand(All)": "EXPAND",
    "Expand(Into)": "EXPINTO",
    "OptionalExpand(All)": "OPTEXP",
    "OptionalExpand(Into)": "OPTEXPINTO",
    "VarLengthExpand(All)": "VAREXP",
    "VarLengthExpand(Into)": "VAREXPINTO",
    "VarLengthExpand(Pruning)": "PRUNEXP",
    
    # Filter and predicates
    "Filter": "FILTER",
    "Argument": "ARG",
    "Selection": "SELECT",
    
    # Aggregation and grouping
    "EagerAggregation": "EAGERAGG",
    "OrderedAggregation": "ORDAGG",
    "Aggregation": "AGG",
    "Distinct": "DISTINCT",
    
    # Sorting and limiting
    "Sort": "SORT",
    "Top": "TOP",
    "Skip": "SKIP",
    "Limit": "LIMIT",
    "PartialSort": "PARTSORT",
    
    # Join operations
    "NodeHashJoin": "HJOIN",
    "ValueHashJoin": "VHJOIN",
    "NodeLeftOuterHashJoin": "LOJOIN",
    "NodeRightOuterHashJoin": "ROJOIN",
    "CartesianProduct": "CROSS",
    "Apply": "APPLY",
    "SemiApply": "SEMIAPPLY",
    "AntiSemiApply": "ANTISEMI",
    "SelectOrSemiApply": "SELORSA",
    "SelectOrAntiSemiApply": "SELORAS",
    "LetSemiApply": "LETSA",
    "LetAntiSemiApply": "LETASA",
    "ConditionalApply": "CONDAPPLY",
    "RollUpApply": "ROLLUP",
    "ForeachApply": "FOREACH",
    
    # Set operations
    "Union": "UNION",
    "OrderedUnion": "ORDUNION",
    
    # Results and projection
    "ProduceResults": "RESULTS",
    "Projection": "PROJECT",
    "Eager": "EAGER",
    "CacheProperties": "CACHE",
    
    # Write operations
    "Create": "CREATE",
    "Merge": "MERGE",
    "Delete": "DELETE",
    "DetachDelete": "DETDELETE",
    "SetProperty": "SETPROP",
    "SetNodeProperty": "SETNPROP",
    "SetRelationshipProperty": "SETRPROP",
    "SetLabels": "SETLABELS",
    "RemoveLabels": "RMLABELS",
    "SetNodePropertiesFromMap": "SETMAP",
    "SetRelationshipPropertiesFromMap": "SETRMAP",
    "CreateNode": "CRNODE",
    "CreateRelationship": "CRREL",
    "EmptyResult": "EMPTY",
    
    # Locking
    "LockNodes": "LOCK",
    
    # Procedures and functions
    "ProcedureCall": "PROC",
    
    # Subqueries
    "SubqueryForeach": "SUBQFE",
    "TransactionForeach": "TXFE",
    "TransactionApply": "TXAPPLY",
    
    # Other
    "LoadCSV": "LOADCSV",
    "Unwind": "UNWIND",
    "Optional": "OPTIONAL",
    "AntiConditionalApply": "ANTICOND",
    "AssertSameNode": "ASSERTN",
    "TriadicSelection": "TRIADIC",
    "TriadicBuild": "TRIBUILD",
    "TriadicFilter": "TRIFILT",
    "Input": "INPUT",
}


def _clean_operator_type(op_type: str) -> str:
    """Remove @neo4j suffix from operator type."""
    # Remove common suffixes like @neo4j
    if "@" in op_type:
        op_type = op_type.split("@")[0]
    return op_type


def _abbreviate_operator(op_type: str) -> str:
    """Create a short abbreviation for operator types not in NODE_ICONS."""
    # Split CamelCase into words and take first letters
    import re
    words = re.findall(r'[A-Z][a-z]*', op_type)
    if words:
        # Take first 2 chars of each word, join them
        abbrev = ''.join(w[:2].upper() for w in words[:4])
        return abbrev if len(abbrev) <= 10 else abbrev[:10]
    return op_type[:10].upper()


def _fmt_stats(node: Dict) -> str:
    """Format execution statistics for a node."""
    parts = []
    
    # Estimated rows (from EXPLAIN)
    args = node.get("arguments", {})
    if "EstimatedRows" in args:
        est = args["EstimatedRows"]
        if isinstance(est, float):
            parts.append(f"est={est:.1f}")
        else:
            parts.append(f"est={est}")
    
    # Actual rows and db hits (from PROFILE)
    if "rows" in node and node["rows"] is not None:
        parts.append(f"rows={node['rows']}")
    if "dbHits" in node and node["dbHits"] is not None:
        parts.append(f"hits={node['dbHits']}")
    
    # Page cache (from PROFILE, if available)
    pc_hits = node.get("pageCacheHits")
    pc_misses = node.get("pageCacheMisses")
    if pc_hits is not None and pc_misses is not None:
        total = pc_hits + pc_misses
        if total > 0:
            ratio = pc_hits / total * 100
            parts.append(f"cache={ratio:.0f}%")
    
    return "  " + f"[{', '.join(parts)}]" if parts else ""


def _node_label(node: Dict) -> str:
    """Generate a descriptive label for a plan node."""
    op_type = _clean_operator_type(node.get("operatorType", "?"))
    icon = NODE_ICONS.get(op_type, _abbreviate_operator(op_type))
    label = icon
    
    extras: List[str] = []
    args = node.get("arguments", {})
    
    # Label for scans
    if "LabelName" in args:
        extras.append(f":{args['LabelName']}")
    
    # Index name
    if "Index" in args:
        extras.append(f"idx={args['Index']}")
    
    # Details (filter, predicate, etc.)
    if "Details" in args:
        details = str(args["Details"])
        if len(details) > 70:
            details = details[:67] + "..."
        extras.append(details)
    
    # Expression
    if "Expression" in args:
        expr = str(args["Expression"])
        if len(expr) > 60:
            expr = expr[:57] + "..."
        extras.append(expr)
    
    # Order 
    if "Order" in args:
        order = str(args["Order"])
        if len(order) > 40:
            order = order[:37] + "..."
        extras.append(f"by={order}")
    
    # Relationship types
    if "RelationshipTypes" in args:
        rel_types = args["RelationshipTypes"]
        if rel_types:
            extras.append(f"rels={rel_types}")
    
    # Identifiers are shown only if extras is empty (to keep label readable)
    identifiers = node.get("identifiers", [])
    if identifiers and not extras:
        id_str = ", ".join(identifiers[:3])
        if len(identifiers) > 3:
            id_str += "..."
        extras.append(f"({id_str})")
    
    if extras:
        label += " " + " ".join(extras)
    
    label += _fmt_stats(node)
    return label


def _render_tree(node: Dict, prefix: str = "", is_last: bool = True) -> List[str]:
    """Recursively render the plan tree into a list of lines."""
    connector = "└─ " if is_last else "├─ "
    lines = [prefix + connector + _node_label(node)]
    
    child_prefix = prefix + ("   " if is_last else "│  ")
    
    children: List[Dict] = node.get("children", [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        lines.extend(_render_tree(child, child_prefix, last))
    
    return lines


def render_plan_tree(plan: Dict) -> str:
    """Return the full visual tree as a single string."""
    header_parts = []
    
    # Summary statistics from PROFILE
    summary = plan.get("_summary", {})
    if "resultAvailableAfter" in summary:
        header_parts.append(f"Planning: {summary['resultAvailableAfter']} ms")
    if "resultConsumedAfter" in summary:
        header_parts.append(f"Execution: {summary['resultConsumedAfter']} ms")
    
    header = ""
    if header_parts:
        header = "  " + " | ".join(header_parts) + "\n\n"
    
    lines = _render_tree(plan)
    return header + "\n".join(lines)


def print_tree(plan: Dict) -> None:
    """Print the plan tree with rich formatting."""
    tree_str = render_plan_tree(plan)
    console.print(Panel(tree_str, title="[bold]Query Plan Tree[/bold]",
                        border_style="blue"))


def print_json(plan: Dict) -> None:
    """Print the plan JSON with syntax highlighting."""
    # Remove internal summary for cleaner JSON output
    plan_copy = {k: v for k, v in plan.items() if not k.startswith("_")}
    json_str = json.dumps(plan_copy, indent=2)
    console.print(Panel(
        Syntax(json_str, "json", theme="monokai", line_numbers=False,
               word_wrap=False),
        title="[bold]Raw JSON Plan[/bold]",
        border_style="green",
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a Neo4j Cypher query plan visually.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Cypher query string. Read from stdin if omitted.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use PROFILE instead of EXPLAIN (actually runs the query; DML is rolled back).",
    )
    parser.add_argument(
        "--no-rollback",
        dest="no_rollback",
        action="store_true",
        help="Do NOT rollback DML operations when using --profile (dangerous!).",
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
        print("Enter your Cypher query (finish with Ctrl-D on a blank line):")
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
    
    is_dml = bool(DML_RE.search(query))
    
    if is_dml:
        if args.profile:
            if args.no_rollback:
                print(
                    "WARNING: DML query will be executed and changes WILL be committed!\n",
                    file=sys.stderr,
                )
            else:
                print(
                    "Note: DML query detected — running inside a transaction "
                    "that will be rolled back (no data will be modified).\n",
                    file=sys.stderr,
                )
        else:
            print(
                "Note: DML query detected — using EXPLAIN without execution "
                "(estimated plan only, query will NOT be executed).\n",
                file=sys.stderr,
            )
    
    try:
        plan = fetch_plan(
            query, 
            profile=args.profile, 
            rollback_dml=not args.no_rollback
        )
    except Exception as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not plan:
        print("Error: empty plan returned.", file=sys.stderr)
        sys.exit(1)
    
    if not args.json_only:
        print_tree(plan)
    
    if not args.tree_only:
        print()
        print_json(plan)


if __name__ == "__main__":
    main()
