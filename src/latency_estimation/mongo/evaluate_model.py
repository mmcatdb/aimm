import torch
import numpy as np
import json
import time
import datetime
import random
import argparse
from typing import Dict, List

from config import MongoConfig
from plan_structured_network import PlanStructuredNetwork
from feature_extractor import FeatureExtractor


class ModelEvaluator:
    """Evaluates a trained MongoDB QPP-Net model."""

    def __init__(self, model_path: str = "mongo_qpp_model.pt",
                 host: str = "localhost", port: int = 27017,
                 dbname: str = "tpch"):
        print("Loading trained model...")
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

        self.feature_extractor = checkpoint["feature_extractor"]
        self.coll_stats = checkpoint["coll_stats"]
        self.feature_extractor.set_collection_stats(self.coll_stats)

        cfg = checkpoint["config"]
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            data_vec_dim=cfg["data_vec_dim"],
        )
        self.model.initialize_units_from_operator_info(checkpoint["operator_info"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if "metrics" in checkpoint:
            m = checkpoint["metrics"]
            print(f"  Training MAE: {m['train']['mae']:.2f} ms")
            print(f"  Test MAE: {m['test']['mae']:.2f} ms")
            print(f"  Test R<=2.0: {m['test']['r_within_2.0']*100:.1f}%")

        self.config = MongoConfig(host=host, port=port, dbname=dbname)
        self.db = self.config.get_db()
        print("Model loaded.\n")

    def predict_from_explain(self, explain_output: Dict) -> float:
        """Predict latency from a queryPlanner explain (no execution)."""
        plan_tree = PlanStructuredNetwork.extract_plan_tree(explain_output)
        collection = PlanStructuredNetwork.get_collection_from_explain(explain_output)
        with torch.no_grad():
            pred = self.model(plan_tree, collection).item()
        return pred

    def evaluate_find(self, collection: str, filter_doc: dict,
                      projection=None, sort=None, limit=0, skip=0,
                      num_runs: int = 5, label: str = "") -> Dict:
        """Evaluate a single find query."""
        # 1. Get plan without executing
        cmd = {"find": collection, "filter": filter_doc}
        if projection: cmd["projection"] = projection
        if sort: cmd["sort"] = sort
        if limit: cmd["limit"] = limit
        if skip: cmd["skip"] = skip

        explain = self.db.command("explain", cmd, verbosity="queryPlanner")
        predicted_ms = self.predict_from_explain(explain)

        # 2. Measure actual execution time
        coll = self.db[collection]
        times = []
        for _ in range(num_runs):
            if projection:
                cursor = coll.find(filter_doc, projection)
            else:
                cursor = coll.find(filter_doc)
            if sort: cursor = cursor.sort(list(sort.items()))
            if skip: cursor = cursor.skip(skip)
            if limit: cursor = cursor.limit(limit)
            start = time.perf_counter()
            list(cursor)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        actual_ms = np.mean(times)
        r_value = max(predicted_ms / (actual_ms + 1e-8),
                      actual_ms / (predicted_ms + 1e-8))

        return {
            "label": label,
            "predicted_ms": predicted_ms,
            "actual_ms": actual_ms,
            "actual_min": np.min(times),
            "actual_max": np.max(times),
            "error_ms": abs(predicted_ms - actual_ms),
            "relative_error": abs(predicted_ms - actual_ms) / (actual_ms + 1e-8),
            "r_value": r_value,
        }

    def evaluate_aggregate(self, collection: str, pipeline: list,
                           num_runs: int = 5, label: str = "") -> Dict:
        """Evaluate a single aggregate query."""
        cmd = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
        explain = self.db.command("explain", cmd, verbosity="queryPlanner")
        predicted_ms = self.predict_from_explain(explain)

        coll = self.db[collection]
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            list(coll.aggregate(pipeline))
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        actual_ms = np.mean(times)
        r_value = max(predicted_ms / (actual_ms + 1e-8),
                      actual_ms / (predicted_ms + 1e-8))

        return {
            "label": label,
            "predicted_ms": predicted_ms,
            "actual_ms": actual_ms,
            "actual_min": np.min(times),
            "actual_max": np.max(times),
            "error_ms": abs(predicted_ms - actual_ms),
            "relative_error": abs(predicted_ms - actual_ms) / (actual_ms + 1e-8),
            "r_value": r_value,
        }


def generate_test_queries() -> List[Dict]:
    """
    Generate test queries

    A) Novel predicate structures ($or, $and, $ne, $nin, $exists, $regex)
    B) Extreme selectivities (near-empty, near-full-table)
    C) Unseen field combinations and date fields (l_commitdate, l_receiptdate)
    D) Different aggregation patterns ($unwind, $project, $count, multi-$group)
    E) Cross-collection lookups not in training (supplier→nation, lineitem→orders)
    F) Large skip values, unlimited scans on medium tables
    G) Multi-stage (4+) aggregate pipelines
    """
    queries = []

    # =====================================================================
    # SECTION A  –  Baseline find queries (kept from original for coverage)
    # =====================================================================

    # A1. Lineitem date ranges
    for delta_days in [30, 90, 180, 365, 730]:
        d = datetime.datetime(1998, 12, 1) - datetime.timedelta(days=delta_days)
        queries.append({
            "type": "find", "collection": "lineitem",
            "params": {"filter": {"l_shipdate": {"$lte": d}}, "limit": 100},
            "label": f"Lineitem shipdate <= {d.strftime('%Y-%m-%d')} LIMIT 100",
        })

    # A2. Orders price thresholds
    for threshold in [50000, 150000, 250000, 350000, 450000]:
        queries.append({
            "type": "find", "collection": "orders",
            "params": {"filter": {"o_totalprice": {"$gt": threshold}},
                       "sort": {"o_totalprice": -1}, "limit": 50},
            "label": f"Orders price>{threshold} sorted LIMIT 50",
        })

    # A3. Customer segment queries
    for seg in ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"]:
        queries.append({
            "type": "find", "collection": "customer",
            "params": {"filter": {"c_mktsegment": seg, "c_acctbal": {"$gt": 5000}},
                       "sort": {"c_acctbal": -1}, "limit": 30},
            "label": f"Customer {seg} acctbal>5000",
        })

    # A4. Part queries
    for brand_num in [13, 24, 35, 42, 51]:
        queries.append({
            "type": "find", "collection": "part",
            "params": {"filter": {"p_brand": f"Brand#{brand_num}",
                                  "p_size": {"$gte": 10, "$lte": 30}},
                       "sort": {"p_retailprice": -1}},
            "label": f"Part Brand#{brand_num} size 10-30",
        })

    # A5. Supplier balance
    for bal in [2000, 4000, 6000, 8000]:
        queries.append({
            "type": "find", "collection": "supplier",
            "params": {"filter": {"s_acctbal": {"$gt": bal}}},
            "label": f"Supplier acctbal>{bal}",
        })

    # A6. Index point lookups
    for key in [42, 1000, 5000, 100000, 250000]:
        queries.append({
            "type": "find", "collection": "orders",
            "params": {"filter": {"o_orderkey": key}},
            "label": f"Orders point lookup key={key}",
        })

    # A7. Multi-key index lookups
    keys = [random.randint(1, 300000) for _ in range(15)]
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderkey": {"$in": keys}},
                   "projection": {"o_custkey": 1, "o_totalprice": 1}},
        "label": "Orders 15-key $in lookup",
    })

    # A8. Full collection scans (tiny tables)
    queries.append({"type": "find", "collection": "nation",
                    "params": {"filter": {}}, "label": "Nation full scan"})
    queries.append({"type": "find", "collection": "region",
                    "params": {"filter": {}}, "label": "Region full scan"})

    # A9. Partsupp sorted
    for cost in [50, 200, 500]:
        queries.append({
            "type": "find", "collection": "partsupp",
            "params": {"filter": {"ps_supplycost": {"$lt": cost}},
                       "sort": {"ps_supplycost": 1}, "limit": 100},
            "label": f"Partsupp cost<{cost} sorted LIMIT 100",
        })

    # A10. Baseline aggregations
    for delta in [60, 90, 120]:
        d = datetime.datetime(1998, 12, 1) - datetime.timedelta(days=delta)
        queries.append({
            "type": "aggregate", "collection": "lineitem",
            "params": {"pipeline": [
                {"$match": {"l_shipdate": {"$lte": d}}},
                {"$group": {"_id": {"rf": "$l_returnflag", "ls": "$l_linestatus"},
                            "sum_qty": {"$sum": "$l_quantity"},
                            "sum_price": {"$sum": "$l_extendedprice"},
                            "avg_disc": {"$avg": "$l_discount"},
                            "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}},
            ]},
            "label": f"TPC-H Q1 delta={delta}d",
        })

    for year in [1994, 1995, 1996, 1997]:
        d = datetime.datetime(year, 1, 1)
        queries.append({
            "type": "aggregate", "collection": "orders",
            "params": {"pipeline": [
                {"$match": {"o_orderdate": {"$gte": d}}},
                {"$group": {"_id": "$o_orderpriority",
                            "count": {"$sum": 1},
                            "avg_price": {"$avg": "$o_totalprice"}}},
                {"$sort": {"count": -1}},
            ]},
            "label": f"Orders group priority year>={year}",
        })

    for price_threshold in [200000, 300000, 400000]:
        queries.append({
            "type": "aggregate", "collection": "orders",
            "params": {"pipeline": [
                {"$match": {"o_totalprice": {"$gt": price_threshold}}},
                {"$limit": 10},
                {"$lookup": {"from": "customer", "localField": "o_custkey",
                             "foreignField": "c_custkey", "as": "customer"}},
            ]},
            "label": f"Orders lookup customer price>{price_threshold}",
        })

    for size_threshold in [20, 30, 40]:
        queries.append({
            "type": "aggregate", "collection": "part",
            "params": {"pipeline": [
                {"$match": {"p_size": {"$gt": size_threshold}}},
                {"$limit": 20},
                {"$lookup": {"from": "partsupp", "localField": "p_partkey",
                             "foreignField": "ps_partkey", "as": "suppliers"}},
            ]},
            "label": f"Part lookup partsupp size>{size_threshold}",
        })

    # =====================================================================
    # SECTION B  –  Novel predicate structures ($or, $and, $ne, $nin, $exists, $regex)
    # =====================================================================

    # B1. $or across different fields (training never uses $or in find)
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"$or": [
            {"o_totalprice": {"$lt": 5000}},
            {"o_totalprice": {"$gt": 450000}},
        ]}, "limit": 200},
        "label": "Orders $or extreme prices",
    })

    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {"$or": [
            {"c_mktsegment": "BUILDING", "c_acctbal": {"$gt": 9000}},
            {"c_mktsegment": "FURNITURE", "c_acctbal": {"$lt": -500}},
        ]}},
        "label": "Customer $or segment+balance",
    })

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"$or": [
            {"l_returnflag": "R"},
            {"l_shipmode": "AIR", "l_quantity": {"$gt": 45}},
        ]}, "limit": 500},
        "label": "Lineitem $or returnflag|shipmode+qty",
    })

    # B2. $and with many predicates (3+ field compound filter)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "$and": [
                {"l_shipdate": {"$gte": datetime.datetime(1995, 1, 1)}},
                {"l_shipdate": {"$lt": datetime.datetime(1996, 1, 1)}},
                {"l_discount": {"$gte": 0.05, "$lte": 0.07}},
                {"l_quantity": {"$lt": 24}},
            ]
        }},
        "label": "Lineitem $and 4-pred (TPC-H Q6 style)",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {
            "$and": [
                {"o_orderdate": {"$gte": datetime.datetime(1994, 1, 1)}},
                {"o_orderdate": {"$lt": datetime.datetime(1995, 1, 1)}},
                {"o_orderpriority": {"$in": ["1-URGENT", "2-HIGH"]}},
                {"o_totalprice": {"$gt": 100000}},
            ]
        }, "limit": 100},
        "label": "Orders $and date+priority+price",
    })

    # B3. $ne operator (never in training)
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderstatus": {"$ne": "F"}},
                   "limit": 200},
        "label": "Orders status $ne F",
    })

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_returnflag": {"$ne": "N"},
                               "l_shipmode": {"$ne": "TRUCK"}},
                   "limit": 300},
        "label": "Lineitem $ne returnflag+shipmode",
    })

    # B4. $nin operator
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_shipmode": {"$nin": ["AIR", "MAIL", "RAIL"]}},
                   "limit": 500},
        "label": "Lineitem shipmode $nin 3 values",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderpriority": {"$nin": ["1-URGENT", "2-HIGH"]}},
                   "limit": 200},
        "label": "Orders priority $nin urgent/high",
    })

    # B5. $exists (structural predicate — never in training)
    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {"c_comment": {"$exists": True},
                               "c_acctbal": {"$gt": 8000}}},
        "label": "Customer $exists comment + high bal",
    })

    # B6. $regex (never in training)
    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {"c_name": {"$regex": "^Customer#00001"}},
                   "limit": 50},
        "label": "Customer $regex name prefix",
    })

    queries.append({
        "type": "find", "collection": "part",
        "params": {"filter": {"p_type": {"$regex": "BRASS$"}}},
        "label": "Part $regex type ends BRASS",
    })

    queries.append({
        "type": "find", "collection": "part",
        "params": {"filter": {"p_name": {"$regex": "green"}}},
        "label": "Part $regex name contains green",
    })

    # B7. Nested $or inside $and
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "$and": [
                {"l_shipdate": {"$gte": datetime.datetime(1994, 1, 1)}},
                {"$or": [
                    {"l_shipmode": "AIR"},
                    {"l_shipmode": "REG AIR"},
                ]},
                {"$or": [
                    {"l_shipinstruct": "DELIVER IN PERSON"},
                    {"l_quantity": {"$lt": 10}},
                ]},
            ]
        }, "limit": 200},
        "label": "Lineitem nested $and/$or (TPC-H Q19 style)",
    })

    # =====================================================================
    # SECTION C  –  Extreme selectivities
    # =====================================================================

    # C1. Near-empty results (very tight filter)
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_totalprice": {"$gt": 490000}}},
        "label": "Orders extreme high price (near empty)",
    })

    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {"c_acctbal": {"$gt": 9990}}},
        "label": "Customer acctbal>9990 (near empty)",
    })

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "l_quantity": 1.0,
            "l_discount": 0.0,
            "l_tax": 0.0,
        }},
        "label": "Lineitem exact qty=1 disc=0 tax=0 (near empty)",
    })

    # C2. Near-full-table scans (very permissive filter)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_quantity": {"$gt": 0}}},
        "label": "Lineitem qty>0 (full table, 1.2M rows)",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_totalprice": {"$gt": 0}}},
        "label": "Orders price>0 (full table, 300K rows)",
    })

    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {}},
        "label": "Customer full scan (30K rows)",
    })

    queries.append({
        "type": "find", "collection": "partsupp",
        "params": {"filter": {}},
        "label": "Partsupp full scan (160K rows)",
    })

    queries.append({
        "type": "find", "collection": "supplier",
        "params": {"filter": {}},
        "label": "Supplier full scan (2K rows)",
    })

    # C3. Unlimited scans with sort on medium table (no limit!)
    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {"c_acctbal": {"$gt": 0}},
                   "sort": {"c_acctbal": -1}},
        "label": "Customer bal>0 sorted NO LIMIT",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderdate": {"$gte": datetime.datetime(1997, 1, 1)}},
                   "sort": {"o_totalprice": -1}},
        "label": "Orders 1997+ sorted by price NO LIMIT",
    })

    # =====================================================================
    # SECTION D  –  Unseen fields & field combinations
    # =====================================================================

    # D1. l_commitdate (never used in training)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "l_commitdate": {"$lt": datetime.datetime(1993, 6, 1)}},
            "limit": 200},
        "label": "Lineitem commitdate<1993-06 LIMIT 200",
    })

    # D2. l_receiptdate (never used in training)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "l_receiptdate": {"$gte": datetime.datetime(1998, 6, 1)}},
            "limit": 500},
        "label": "Lineitem receiptdate>=1998-06",
    })

    # D3. Cross-date comparison style: shipdate AND commitdate
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "l_shipdate": {"$gte": datetime.datetime(1995, 1, 1)},
            "l_commitdate": {"$lt": datetime.datetime(1995, 3, 1)},
            "l_receiptdate": {"$gte": datetime.datetime(1995, 2, 1)},
        }, "limit": 200},
        "label": "Lineitem 3-date cross filter",
    })

    # D4. Orders by orderstatus (categorical, not used much in training)
    for status in ["F", "O", "P"]:
        queries.append({
            "type": "find", "collection": "orders",
            "params": {"filter": {"o_orderstatus": status}, "limit": 500},
            "label": f"Orders status={status} LIMIT 500",
        })

    # D5. Orders by o_clerk (high cardinality string, never in training)
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_clerk": "Clerk#000000951"}},
        "label": "Orders exact clerk lookup",
    })

    # D6. Lineitem by ship instruction + mode (string equality combo)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {
            "l_shipinstruct": "DELIVER IN PERSON",
            "l_shipmode": "AIR",
        }, "limit": 300},
        "label": "Lineitem shipinstruct+mode combo",
    })

    # D7. Part by container + type (string fields not in training)
    queries.append({
        "type": "find", "collection": "part",
        "params": {"filter": {
            "p_container": {"$in": ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]},
            "p_size": {"$lte": 5},
        }},
        "label": "Part small containers size<=5",
    })

    queries.append({
        "type": "find", "collection": "part",
        "params": {"filter": {"p_mfgr": "Manufacturer#3",
                               "p_type": {"$regex": "^ECONOMY"}}},
        "label": "Part Mfgr#3 ECONOMY types",
    })

    # D8. Customer by nationkey (joining field, never filtered in training)
    for nk in [0, 7, 15, 24]:
        queries.append({
            "type": "find", "collection": "customer",
            "params": {"filter": {"c_nationkey": nk}},
            "label": f"Customer by nationkey={nk}",
        })

    # D9. Partsupp by availqty ranges not in training
    queries.append({
        "type": "find", "collection": "partsupp",
        "params": {"filter": {"ps_availqty": {"$lt": 100}}},
        "label": "Partsupp very low avail qty<100",
    })

    queries.append({
        "type": "find", "collection": "partsupp",
        "params": {"filter": {"ps_availqty": {"$gt": 9500},
                               "ps_supplycost": {"$gt": 900}}},
        "label": "Partsupp high qty+cost",
    })

    # =====================================================================
    # SECTION E  –  Large skip / pagination patterns
    # =====================================================================

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_totalprice": {"$gt": 100000}},
                   "sort": {"o_totalprice": -1},
                   "skip": 5000, "limit": 50},
        "label": "Orders deep pagination skip=5000",
    })

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_shipdate": {"$gte": datetime.datetime(1997, 1, 1)}},
                   "skip": 10000, "limit": 100},
        "label": "Lineitem deep skip=10000",
    })

    queries.append({
        "type": "find", "collection": "customer",
        "params": {"filter": {},
                   "sort": {"c_acctbal": -1},
                   "skip": 15000, "limit": 50},
        "label": "Customer skip=15000 sorted",
    })

    # =====================================================================
    # SECTION F  –  $in on large value lists (varying list sizes)
    # =====================================================================

    # Large $in list on indexed field
    keys_50 = [random.randint(1, 300000) for _ in range(50)]
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderkey": {"$in": keys_50}}},
        "label": "Orders 50-key $in lookup",
    })

    keys_200 = [random.randint(1, 300000) for _ in range(200)]
    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_orderkey": {"$in": keys_200}}},
        "label": "Orders 200-key $in lookup",
    })

    # $in on non-indexed field (string values)
    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_shipmode": {"$in": ["AIR", "RAIL"]}},
                   "limit": 1000},
        "label": "Lineitem shipmode $in [AIR,RAIL]",
    })

    queries.append({
        "type": "find", "collection": "part",
        "params": {"filter": {"p_brand": {"$in": [f"Brand#{i}" for i in range(11, 20)]}}},
        "label": "Part 9-brand $in lookup",
    })

    # =====================================================================
    # SECTION G  –  Novel aggregation patterns
    # =====================================================================

    # G1. $count stage
    queries.append({
        "type": "aggregate", "collection": "lineitem",
        "params": {"pipeline": [
            {"$match": {"l_shipmode": "AIR",
                        "l_shipdate": {"$gte": datetime.datetime(1997, 1, 1)}}},
            {"$count": "total"},
        ]},
        "label": "Lineitem $count AIR 1997+",
    })

    queries.append({
        "type": "aggregate", "collection": "orders",
        "params": {"pipeline": [
            {"$match": {"o_orderstatus": "P"}},
            {"$count": "total"},
        ]},
        "label": "Orders $count status=P",
    })

    # G2. $group with $min/$max/$first/$last (accumulators not in training)
    queries.append({
        "type": "aggregate", "collection": "lineitem",
        "params": {"pipeline": [
            {"$match": {"l_shipdate": {"$gte": datetime.datetime(1996, 1, 1)}}},
            {"$group": {
                "_id": "$l_shipmode",
                "min_price": {"$min": "$l_extendedprice"},
                "max_price": {"$max": "$l_extendedprice"},
                "first_date": {"$first": "$l_shipdate"},
                "last_date": {"$last": "$l_shipdate"},
                "count": {"$sum": 1},
            }},
        ]},
        "label": "Lineitem group shipmode min/max/first/last",
    })

    # G3. $group by high-cardinality field (o_custkey → many groups)
    queries.append({
        "type": "aggregate", "collection": "orders",
        "params": {"pipeline": [
            {"$group": {
                "_id": "$o_custkey",
                "order_count": {"$sum": 1},
                "total_spent": {"$sum": "$o_totalprice"},
            }},
            {"$sort": {"total_spent": -1}},
            {"$limit": 20},
        ]},
        "label": "Orders group by custkey top-20 spenders",
    })

    # G4. $project stage (reshaping, never in training pipelines)
    queries.append({
        "type": "aggregate", "collection": "lineitem",
        "params": {"pipeline": [
            {"$match": {"l_shipdate": {"$gte": datetime.datetime(1997, 6, 1)}}},
            {"$project": {
                "revenue": {"$multiply": [
                    "$l_extendedprice",
                    {"$subtract": [1, "$l_discount"]},
                ]},
                "l_shipmode": 1,
                "l_returnflag": 1,
            }},
            {"$group": {
                "_id": "$l_shipmode",
                "total_revenue": {"$sum": "$revenue"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"total_revenue": -1}},
        ]},
        "label": "Lineitem $project revenue then group (4 stages)",
    })

    # G5. Customer aggregation by segment with $project (never in training)
    queries.append({
        "type": "aggregate", "collection": "customer",
        "params": {"pipeline": [
            {"$match": {"c_acctbal": {"$gt": 0}}},
            {"$group": {
                "_id": "$c_mktsegment",
                "avg_bal": {"$avg": "$c_acctbal"},
                "max_bal": {"$max": "$c_acctbal"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"avg_bal": -1}},
        ]},
        "label": "Customer group by segment avg/max bal",
    })

    # G6. Multi-stage pipeline with $match → $group → $match → $sort (double filter)
    queries.append({
        "type": "aggregate", "collection": "orders",
        "params": {"pipeline": [
            {"$match": {"o_orderdate": {"$gte": datetime.datetime(1993, 1, 1),
                                         "$lt": datetime.datetime(1998, 1, 1)}}},
            {"$group": {
                "_id": "$o_custkey",
                "order_count": {"$sum": 1},
                "avg_price": {"$avg": "$o_totalprice"},
            }},
            {"$match": {"order_count": {"$gte": 10}}},
            {"$sort": {"avg_price": -1}},
            {"$limit": 50},
        ]},
        "label": "Orders double-$match group then filter (TPC-H Q13 style)",
    })

    # G7. Partsupp aggregation (collection rarely aggregated in training)
    queries.append({
        "type": "aggregate", "collection": "partsupp",
        "params": {"pipeline": [
            {"$match": {"ps_supplycost": {"$lt": 100}}},
            {"$group": {
                "_id": "$ps_suppkey",
                "part_count": {"$sum": 1},
                "avg_cost": {"$avg": "$ps_supplycost"},
                "total_qty": {"$sum": "$ps_availqty"},
            }},
            {"$sort": {"part_count": -1}},
            {"$limit": 20},
        ]},
        "label": "Partsupp group by supplier cost<100",
    })

    # G8. Supplier aggregation by nation
    queries.append({
        "type": "aggregate", "collection": "supplier",
        "params": {"pipeline": [
            {"$group": {
                "_id": "$s_nationkey",
                "supplier_count": {"$sum": 1},
                "avg_balance": {"$avg": "$s_acctbal"},
            }},
            {"$sort": {"avg_balance": -1}},
        ]},
        "label": "Supplier group by nationkey",
    })

    # =====================================================================
    # SECTION H  –  Novel cross-collection lookups
    # =====================================================================

    # H1. Supplier → nation (never in training)
    queries.append({
        "type": "aggregate", "collection": "supplier",
        "params": {"pipeline": [
            {"$match": {"s_acctbal": {"$gt": 5000}}},
            {"$lookup": {"from": "nation", "localField": "s_nationkey",
                         "foreignField": "n_nationkey", "as": "nation_info"}},
        ]},
        "label": "Supplier lookup nation bal>5000",
    })

    # H2. Customer → nation → region chain (2 lookups, never in training)
    queries.append({
        "type": "aggregate", "collection": "customer",
        "params": {"pipeline": [
            {"$match": {"c_acctbal": {"$gt": 9000}}},
            {"$limit": 50},
            {"$lookup": {"from": "nation", "localField": "c_nationkey",
                         "foreignField": "n_nationkey", "as": "nation"}},
            {"$unwind": "$nation"},
            {"$lookup": {"from": "region", "localField": "nation.n_regionkey",
                         "foreignField": "r_regionkey", "as": "region"}},
        ]},
        "label": "Customer→nation→region chain (5 stages)",
    })

    # H3. Lineitem → orders lookup (largest collection as driving table)
    queries.append({
        "type": "aggregate", "collection": "lineitem",
        "params": {"pipeline": [
            {"$match": {"l_shipdate": {"$gte": datetime.datetime(1998, 11, 1)},
                        "l_returnflag": "N"}},
            {"$limit": 50},
            {"$lookup": {"from": "orders", "localField": "l_orderkey",
                         "foreignField": "o_orderkey", "as": "order_info"}},
        ]},
        "label": "Lineitem→orders lookup recent shipments",
    })

    # H4. Nation → region lookup (tiny→tiny, very different cost profile)
    queries.append({
        "type": "aggregate", "collection": "nation",
        "params": {"pipeline": [
            {"$lookup": {"from": "region", "localField": "n_regionkey",
                         "foreignField": "r_regionkey", "as": "region"}},
            {"$unwind": "$region"},
        ]},
        "label": "Nation→region lookup+unwind (tiny tables)",
    })

    # H5. Orders → customer lookup with group after (lookup + aggregate)
    queries.append({
        "type": "aggregate", "collection": "orders",
        "params": {"pipeline": [
            {"$match": {"o_orderdate": {"$gte": datetime.datetime(1997, 1, 1)}}},
            {"$limit": 100},
            {"$lookup": {"from": "customer", "localField": "o_custkey",
                         "foreignField": "c_custkey", "as": "cust"}},
            {"$unwind": "$cust"},
            {"$group": {
                "_id": "$cust.c_mktsegment",
                "order_count": {"$sum": 1},
                "avg_price": {"$avg": "$o_totalprice"},
            }},
            {"$sort": {"order_count": -1}},
        ]},
        "label": "Orders→customer lookup+unwind+group (6 stages)",
    })

    # =====================================================================
    # SECTION I  –  $unwind-heavy pipelines
    # =====================================================================

    # I1. Part→partsupp lookup + unwind + group (TPC-H Q2/Q11 style)
    queries.append({
        "type": "aggregate", "collection": "part",
        "params": {"pipeline": [
            {"$match": {"p_size": 15, "p_type": {"$regex": "BRASS$"}}},
            {"$lookup": {"from": "partsupp", "localField": "p_partkey",
                         "foreignField": "ps_partkey", "as": "ps"}},
            {"$unwind": "$ps"},
            {"$sort": {"ps.ps_supplycost": 1}},
            {"$limit": 20},
        ]},
        "label": "Part BRASS size=15 lookup+unwind+sort (TPC-H Q2 style)",
    })

    # I2. Customer → orders lookup + unwind (never in training)
    queries.append({
        "type": "aggregate", "collection": "customer",
        "params": {"pipeline": [
            {"$match": {"c_mktsegment": "AUTOMOBILE"}},
            {"$limit": 20},
            {"$lookup": {"from": "orders", "localField": "c_custkey",
                         "foreignField": "o_custkey", "as": "orders"}},
            {"$unwind": "$orders"},
            {"$group": {
                "_id": "$c_custkey",
                "num_orders": {"$sum": 1},
                "total_spent": {"$sum": "$orders.o_totalprice"},
            }},
            {"$sort": {"total_spent": -1}},
            {"$limit": 10},
        ]},
        "label": "Customer AUTOMOBILE→orders unwind+group (7 stages)",
    })

    # =====================================================================
    # SECTION J  –  Varying limit values (very large, very small, absent)
    # =====================================================================

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_shipdate": {"$gte": datetime.datetime(1995, 1, 1)}},
                   "sort": {"l_extendedprice": -1}, "limit": 1},
        "label": "Lineitem top-1 most expensive 1995+",
    })

    queries.append({
        "type": "find", "collection": "lineitem",
        "params": {"filter": {"l_shipdate": {"$gte": datetime.datetime(1995, 1, 1)}},
                   "sort": {"l_extendedprice": -1}, "limit": 5000},
        "label": "Lineitem top-5000 most expensive 1995+",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_totalprice": {"$gt": 100000}},
                   "sort": {"o_orderdate": 1}, "limit": 1},
        "label": "Orders earliest order price>100K (limit 1)",
    })

    queries.append({
        "type": "find", "collection": "orders",
        "params": {"filter": {"o_totalprice": {"$gt": 100000}},
                   "sort": {"o_orderdate": 1}, "limit": 10000},
        "label": "Orders price>100K sorted by date limit 10000",
    })

    return queries


def main():
    parser = argparse.ArgumentParser(description="Evaluate MongoDB QPP-Net")
    parser.add_argument("--model", type=str, default="mongo_qpp_model.pt")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=27017)
    parser.add_argument("--dbname", type=str, default="tpch")
    parser.add_argument("--save", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("MongoDB QPP-Net Model Evaluation")
    print("=" * 80)

    evaluator = ModelEvaluator(args.model, args.host, args.port, args.dbname)

    queries = generate_test_queries()
    print(f"\nEvaluating {len(queries)} test queries...\n")

    results = []
    for i, q in enumerate(queries):
        try:
            if q["type"] == "find":
                r = evaluator.evaluate_find(
                    q["collection"],
                    q["params"]["filter"],
                    projection=q["params"].get("projection"),
                    sort=q["params"].get("sort"),
                    limit=q["params"].get("limit", 0),
                    skip=q["params"].get("skip", 0),
                    num_runs=args.runs,
                    label=q["label"],
                )
            else:
                r = evaluator.evaluate_aggregate(
                    q["collection"],
                    q["params"]["pipeline"],
                    num_runs=args.runs,
                    label=q["label"],
                )
            results.append(r)
            status = "OK" if r["r_value"] <= 2.0 else ("WARN" if r["r_value"] <= 5.0 else "BAD")
            print(f"  [{status:4s}] {r['label'][:55]:55s}  "
                  f"pred={r['predicted_ms']:8.1f}ms  actual={r['actual_ms']:8.1f}ms  "
                  f"R={r['r_value']:.2f}")
        except Exception as e:
            print(f"  [ERR ] {q['label'][:55]:55s}  {e}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)

        errors = [r["error_ms"] for r in results]
        rel_errors = [r["relative_error"] for r in results]
        r_values = [r["r_value"] for r in results]
        predicted = [rpredicted["predicted_ms"] for r in results]
        actual = [r["actual_ms"] for r in results]

        print(f"  Queries evaluated: {len(results)}")
        print(f"  Mean Absolute Error: {np.mean(errors):.2f} ms")
        print(f"  Median Absolute Error: {np.median(errors):.2f} ms")
        print(f"  Mean Relative Error: {np.mean(rel_errors):.4f}")
        print(f"  Median R-value (Q-error): {np.median(r_values):.2f}")
        print(f"  Mean R-value: {np.mean(r_values):.2f}")
        print(f"  R <= 1.5: {np.mean([r <= 1.5 for r in r_values])*100:.1f}%")
        print(f"  R <= 2.0: {np.mean([r <= 2.0 for r in r_values])*100:.1f}%")
        print(f"  R <= 5.0: {np.mean([r <= 5.0 for r in r_values])*100:.1f}%")

        # Save results
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
