import time
import random
import datetime
from typing import List, Dict, Tuple
from config import MongoConfig


class PlanExtractor:
    """Extracts query plans and execution statistics from MongoDB."""

    def __init__(self, config: MongoConfig):
        self.config = config
        self.db = config.get_db()

    # ------------------------------------------------------------------
    # Explain helpers
    # ------------------------------------------------------------------

    def explain_find(self, collection_name: str, filter_doc: dict,
                     projection=None, sort=None, limit=0, skip=0,
                     verbosity="queryPlanner") -> Dict:
        """Run explain on a find command."""
        cmd = {"find": collection_name, "filter": filter_doc}
        if projection:
            cmd["projection"] = projection
        if sort:
            cmd["sort"] = sort
        if limit:
            cmd["limit"] = limit
        if skip:
            cmd["skip"] = skip
        return self.db.command("explain", cmd, verbosity=verbosity)

    def explain_aggregate(self, collection_name: str, pipeline: list,
                          verbosity="queryPlanner") -> Dict:
        """Run explain on an aggregate pipeline."""
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        return self.db.command("explain", cmd, verbosity=verbosity)

    def execute_find_timed(self, collection_name: str, filter_doc: dict,
                           projection=None, sort=None, limit=0, skip=0,
                           num_runs: int = 3) -> Tuple[float, float, float]:
        """
        Execute a find query and measure wall-clock time.
        Returns (mean_ms, min_ms, max_ms).
        """
        coll = self.db[collection_name]
        times = []
        for _ in range(num_runs):
            cursor = coll.find(filter_doc)
            if projection:
                cursor = coll.find(filter_doc, projection)
            if sort:
                cursor = cursor.sort(list(sort.items()))
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            start = time.perf_counter()
            list(cursor)  # force materialization
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        return sum(times) / len(times), min(times), max(times)

    def execute_aggregate_timed(self, collection_name: str, pipeline: list,
                                num_runs: int = 3) -> Tuple[float, float, float]:
        """Execute an aggregate pipeline and measure wall-clock time."""
        coll = self.db[collection_name]
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            list(coll.aggregate(pipeline))
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        return sum(times) / len(times), min(times), max(times)

    # ------------------------------------------------------------------
    # TPC-H query generation for MongoDB
    # ------------------------------------------------------------------

    def _random_date(self, start_year=1992, end_year=1998) -> datetime.datetime:
        y = random.randint(start_year, end_year)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        return datetime.datetime(y, m, d)

    def generate_training_queries(self, num_queries: int = 500) -> List[Dict]:
        """
        Generate a diverse set of MongoDB queries for training.

        Each item is a dict with:
          type: 'find' or 'aggregate'
          collection: str
          params: dict  (filter, projection, sort, limit, skip for find; pipeline for aggregate)
          label: str   (human-readable description)
        """
        queries = []
        target = num_queries

        # ------ Category 1: Simple finds with filters ------
        n = target // 10
        for _ in range(n):
            # lineitem date range with various operators and limits
            d = self._random_date(1992, 1998)
            op = random.choice(["$gte", "$lte", "$gt", "$lt"])
            queries.append({
                "type": "find",
                "collection": "lineitem",
                "params": {
                    "filter": {"l_shipdate": {op: d}},
                    "limit": random.choice([0, 10, 50, 100, 500, 1000]),
                },
                "label": "lineitem shipdate filter",
            })

        # Lineitem with quantity / price filters (more selectivity diversity)
        for _ in range(n // 2):
            qty = random.uniform(1, 50)
            queries.append({
                "type": "find",
                "collection": "lineitem",
                "params": {
                    "filter": {"l_quantity": {"$gt": qty}},
                    "limit": random.choice([0, 50, 200]),
                },
                "label": "lineitem quantity filter",
            })

        for _ in range(n):
            # orders price range
            lo = random.uniform(50000, 300000)
            hi = lo + random.uniform(10000, 200000)
            queries.append({
                "type": "find",
                "collection": "orders",
                "params": {
                    "filter": {"o_totalprice": {"$gte": lo, "$lte": hi}},
                    "limit": random.choice([0, 20, 50, 200]),
                },
                "label": "orders price range",
            })

        # Orders with single price threshold + sort + limit
        # (matches evaluation pattern exactly)
        for _ in range(n):
            threshold = random.uniform(10000, 500000)
            queries.append({
                "type": "find",
                "collection": "orders",
                "params": {
                    "filter": {"o_totalprice": {"$gt": threshold}},
                    "sort": {"o_totalprice": random.choice([-1, 1])},
                    "limit": random.choice([10, 20, 50, 100]),
                },
                "label": "orders price threshold sorted",
            })

        # ------ Category 2: Index scans (point lookups, small ranges) ------
        for _ in range(n):
            key = random.randint(1, 300000)
            queries.append({
                "type": "find",
                "collection": "orders",
                "params": {
                    "filter": {"o_orderkey": key},
                },
                "label": "orders point lookup",
            })

        for _ in range(n):
            keys = [random.randint(1, 300000) for _ in range(random.randint(2, 20))]
            queries.append({
                "type": "find",
                "collection": "orders",
                "params": {
                    "filter": {"o_orderkey": {"$in": keys}},
                    "projection": {"o_custkey": 1, "o_totalprice": 1},
                },
                "label": "orders multi-key lookup",
            })

        # ------ Category 3: Finds with sort ------
        for _ in range(n):
            d = self._random_date(1995, 1998)
            queries.append({
                "type": "find",
                "collection": "orders",
                "params": {
                    "filter": {"o_orderdate": {"$gte": d}},
                    "sort": {"o_totalprice": random.choice([-1, 1])},
                    "limit": random.choice([10, 20, 50, 100]),
                },
                "label": "orders sorted by price",
            })

        for _ in range(n):
            d = self._random_date(1994, 1997)
            queries.append({
                "type": "find",
                "collection": "lineitem",
                "params": {
                    "filter": {"l_shipdate": {"$gte": d}},
                    "sort": {"l_extendedprice": -1},
                    "limit": random.choice([10, 50, 100, 200]),
                },
                "label": "lineitem sorted by price",
            })

        # ------ Category 4: Aggregation with $group ------
        for _ in range(n):
            d = self._random_date(1994, 1998)
            queries.append({
                "type": "aggregate",
                "collection": "lineitem",
                "params": {
                    "pipeline": [
                        {"$match": {"l_shipdate": {"$gte": d}}},
                        {"$group": {
                            "_id": "$l_returnflag",
                            "count": {"$sum": 1},
                            "avg_qty": {"$avg": "$l_quantity"},
                            "sum_price": {"$sum": "$l_extendedprice"},
                        }},
                    ],
                },
                "label": "lineitem group by returnflag",
            })

        for _ in range(n):
            d = self._random_date(1994, 1997)
            queries.append({
                "type": "aggregate",
                "collection": "orders",
                "params": {
                    "pipeline": [
                        {"$match": {"o_orderdate": {"$gte": d}}},
                        {"$group": {
                            "_id": "$o_orderpriority",
                            "count": {"$sum": 1},
                            "avg_price": {"$avg": "$o_totalprice"},
                        }},
                        {"$sort": {"count": -1}},
                    ],
                },
                "label": "orders group by priority",
            })

        # ------ Category 5: Aggregation with $lookup (join) ------
        for _ in range(n // 2):
            lo = random.uniform(100000, 400000)
            queries.append({
                "type": "aggregate",
                "collection": "orders",
                "params": {
                    "pipeline": [
                        {"$match": {"o_totalprice": {"$gt": lo}}},
                        {"$limit": random.choice([5, 10, 20, 50])},
                        {"$lookup": {
                            "from": "customer",
                            "localField": "o_custkey",
                            "foreignField": "c_custkey",
                            "as": "customer",
                        }},
                    ],
                },
                "label": "orders lookup customer",
            })

        for _ in range(n // 2):
            keys = random.sample(range(1, 40001), random.randint(5, 50))
            queries.append({
                "type": "aggregate",
                "collection": "part",
                "params": {
                    "pipeline": [
                        {"$match": {"p_partkey": {"$in": keys}}},
                        {"$lookup": {
                            "from": "partsupp",
                            "localField": "p_partkey",
                            "foreignField": "ps_partkey",
                            "as": "suppliers",
                        }},
                    ],
                },
                "label": "part lookup partsupp",
            })

        # ------ Category 6: Customer queries ------
        for _ in range(n):
            bal = random.uniform(-1000, 9000)
            seg = random.choice(["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"])
            queries.append({
                "type": "find",
                "collection": "customer",
                "params": {
                    "filter": {"c_acctbal": {"$gt": bal}, "c_mktsegment": seg},
                    "sort": {"c_acctbal": -1},
                    "limit": random.choice([0, 30, 50, 100]),
                },
                "label": "customer segment balance",
            })
        # Customer acctbal only (varying selectivity)
        for _ in range(n // 2):
            bal = random.uniform(-1000, 9500)
            queries.append({
                "type": "find",
                "collection": "customer",
                "params": {
                    "filter": {"c_acctbal": {"$gt": bal}},
                    "limit": random.choice([0, 50, 100]),
                },
                "label": "customer balance only",
            })

        # ------ Category 7: Part queries ------
        for _ in range(n // 2):
            lo = random.randint(1, 40)
            hi = lo + random.randint(5, 20)
            brand = f"Brand#{random.randint(11, 55)}"
            queries.append({
                "type": "find",
                "collection": "part",
                "params": {
                    "filter": {"p_size": {"$gte": lo, "$lte": hi}, "p_brand": brand},
                    "sort": {"p_retailprice": -1},
                    "limit": random.choice([0, 50, 100]),
                },
                "label": "part brand size filter",
            })

        # ------ Category 8: Supplier / small collection scans ------
        # Much more diversity in selectivity for supplier
        for _ in range(n):
            bal = random.uniform(0, 10000)
            queries.append({
                "type": "find",
                "collection": "supplier",
                "params": {
                    "filter": {"s_acctbal": {"$gt": bal}},
                    "sort": {"s_acctbal": -1},
                },
                "label": "supplier balance filter",
            })
        for _ in range(n // 2):
            bal_lo = random.uniform(-1000, 5000)
            bal_hi = bal_lo + random.uniform(500, 5000)
            queries.append({
                "type": "find",
                "collection": "supplier",
                "params": {
                    "filter": {"s_acctbal": {"$gte": bal_lo, "$lte": bal_hi}},
                },
                "label": "supplier balance range",
            })
        for _ in range(n // 4):
            nation_key = random.randint(0, 24)
            queries.append({
                "type": "find",
                "collection": "supplier",
                "params": {
                    "filter": {"s_nationkey": nation_key},
                },
                "label": "supplier by nation",
            })

        # ------ Category 9: Full collection scans (no filter) ------
        for _ in range(n // 4):
            queries.append({
                "type": "find",
                "collection": "nation",
                "params": {"filter": {}},
                "label": "nation full scan",
            })

        for _ in range(n // 4):
            queries.append({
                "type": "find",
                "collection": "region",
                "params": {"filter": {}},
                "label": "region full scan",
            })

        # ------ Category 10: Skip queries ------
        for _ in range(n // 2):
            queries.append({
                "type": "find",
                "collection": "lineitem",
                "params": {
                    "filter": {"l_shipdate": {"$gte": self._random_date(1996, 1998)}},
                    "skip": random.randint(10, 1000),
                    "limit": random.choice([10, 50, 100]),
                },
                "label": "lineitem skip+limit",
            })

        # ------ Category 11: Projection-only queries ------
        for _ in range(n // 2):
            queries.append({
                "type": "find",
                "collection": "customer",
                "params": {
                    "filter": {"c_custkey": random.randint(1, 30000)},
                    "projection": {"c_name": 1, "c_acctbal": 1, "_id": 0},
                },
                "label": "customer projection lookup",
            })

        # ------ Category 12: Complex aggregates with $group + $sort ------
        # TPC-H Q1 variants with varying date ranges (from narrow to full-table scan)
        for _ in range(n):
            d = self._random_date(1992, 1998)
            queries.append({
                "type": "aggregate",
                "collection": "lineitem",
                "params": {
                    "pipeline": [
                        {"$match": {"l_shipdate": {"$gte": d}}},
                        {"$group": {
                            "_id": {"flag": "$l_returnflag", "status": "$l_linestatus"},
                            "sum_qty": {"$sum": "$l_quantity"},
                            "sum_price": {"$sum": "$l_extendedprice"},
                            "avg_disc": {"$avg": "$l_discount"},
                            "count": {"$sum": 1},
                        }},
                        {"$sort": {"sum_price": -1}},
                    ],
                },
                "label": "lineitem TPC-H Q1 style",
            })

        # TPC-H Q1 with $lte (matching evaluation style)
        for _ in range(n):
            d = self._random_date(1994, 1998)
            queries.append({
                "type": "aggregate",
                "collection": "lineitem",
                "params": {
                    "pipeline": [
                        {"$match": {"l_shipdate": {"$lte": d}}},
                        {"$group": {
                            "_id": {"rf": "$l_returnflag", "ls": "$l_linestatus"},
                            "sum_qty": {"$sum": "$l_quantity"},
                            "sum_price": {"$sum": "$l_extendedprice"},
                            "avg_disc": {"$avg": "$l_discount"},
                            "count": {"$sum": 1},
                        }},
                        {"$sort": {"_id": 1}},
                    ],
                },
                "label": "lineitem TPC-H Q1 lte",
            })

        # Simple lineitem aggregation (count + sum only, less accumulators)
        for _ in range(n // 2):
            d = self._random_date(1994, 1998)
            queries.append({
                "type": "aggregate",
                "collection": "lineitem",
                "params": {
                    "pipeline": [
                        {"$match": {"l_shipdate": {"$gte": d}}},
                        {"$group": {
                            "_id": "$l_returnflag",
                            "count": {"$sum": 1},
                        }},
                    ],
                },
                "label": "lineitem simple group",
            })

        # Add partsupp queries (multiple patterns including just cost filter)
        for _ in range(n // 2):
            cost = random.uniform(10, 500)
            qty = random.randint(100, 9000)
            queries.append({
                "type": "find",
                "collection": "partsupp",
                "params": {
                    "filter": {"ps_supplycost": {"$lt": cost}, "ps_availqty": {"$gt": qty}},
                    "sort": {"ps_supplycost": 1},
                    "limit": random.choice([0, 50, 100, 200]),
                },
                "label": "partsupp cost/qty filter",
            })

        # Partsupp with just cost filter + sort (matches evaluation)
        for _ in range(n):
            cost = random.uniform(10, 1000)
            queries.append({
                "type": "find",
                "collection": "partsupp",
                "params": {
                    "filter": {"ps_supplycost": {"$lt": cost}},
                    "sort": {"ps_supplycost": 1},
                    "limit": random.choice([0, 50, 100, 200]),
                },
                "label": "partsupp cost sorted",
            })

        # Supplier without sort (matches evaluation pattern)
        for _ in range(n // 2):
            bal = random.uniform(-1000, 10000)
            queries.append({
                "type": "find",
                "collection": "supplier",
                "params": {
                    "filter": {"s_acctbal": {"$gt": bal}},
                },
                "label": "supplier balance nosort",
            })

        random.shuffle(queries)
        return queries[:num_queries]

    # ------------------------------------------------------------------
    # Training data collection
    # ------------------------------------------------------------------

    def collect_training_data(self, num_queries: int = 500,
                              num_runs: int = 3) -> List[Dict]:
        """
        Collect training dataset: explain plans + actual execution times.

        For each query, we collect:
        - explain('executionStats') for the plan tree + per-stage stats
        - Wall-clock execution time averaged over num_runs

        Returns list of dicts with keys:
            explain, plan_tree, collection, execution_time_ms, label
        """
        query_specs = self.generate_training_queries(num_queries)
        dataset = []

        print(f"Collecting {len(query_specs)} query plans with execution stats...")

        for i, spec in enumerate(query_specs):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(query_specs)} ({100*i//len(query_specs)}%)")

            try:
                coll_name = spec["collection"]
                params = spec["params"]

                if spec["type"] == "find":
                    explain = self.explain_find(
                        coll_name,
                        params["filter"],
                        projection=params.get("projection"),
                        sort=params.get("sort"),
                        limit=params.get("limit", 0),
                        skip=params.get("skip", 0),
                        verbosity="executionStats",
                    )
                    mean_ms, _, _ = self.execute_find_timed(
                        coll_name,
                        params["filter"],
                        projection=params.get("projection"),
                        sort=params.get("sort"),
                        limit=params.get("limit", 0),
                        skip=params.get("skip", 0),
                        num_runs=num_runs,
                    )
                else:
                    explain = self.explain_aggregate(
                        coll_name,
                        params["pipeline"],
                        verbosity="executionStats",
                    )
                    mean_ms, _, _ = self.execute_aggregate_timed(
                        coll_name,
                        params["pipeline"],
                        num_runs=num_runs,
                    )

                dataset.append({
                    "explain": explain,
                    "collection": coll_name,
                    "execution_time_ms": mean_ms,
                    "label": spec["label"],
                    "spec": spec,
                })

            except Exception as e:
                print(f"  Error on query {i} ({spec['label']}): {e}")
                continue

        print(f"\nCollected {len(dataset)} samples successfully")
        return dataset
