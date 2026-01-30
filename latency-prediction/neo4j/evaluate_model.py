"""
Evaluation script for the trained Neo4j Query Performance Predictor.

This script:
1. Loads a trained model
2. Generates new test queries
3. Predicts their latency using EXPLAIN (no execution)
4. Measures actual execution time
5. Compares predictions to actual times
"""
import torch
import numpy as np
import time
import pickle
import argparse
from tabulate import tabulate

from common.config import Config
from common.databases import Neo4j, cypher
from plan_extractor import PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained Neo4j QPP model."""

    def __init__(self, neo4j: Neo4j, checkpoint_path: str, feature_extractor_path: str, device: str = 'cpu', num_layers: int = 10, hidden_dim: int = 128):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            feature_extractor_path: Path to feature extractor pickle file
            device: 'cpu' or 'cuda'
            num_layers: Number of layers in neural units
            hidden_dim: Hidden dimension of neural units
        """
        self.extractor = PlanExtractor(neo4j)
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Load feature extractor
        print('Loading feature extractor...')
        with open(feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)

        # Load checkpoint first to get operator info
        print('Loading model checkpoint...')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create model
        print('Creating model architecture...')
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=32
        )

        # Pre-create neural units based on the checkpoint's state dict keys
        # This ensures all units exist before loading weights
        print('Initializing neural units from checkpoint...')
        state_dict = checkpoint['model_state_dict']

        # Extract unique unit keys from state dict
        unit_keys = set()
        for key in state_dict.keys():
            if key.startswith('units.'):
                # Extract the unit key (e.g., 'ProduceResults_1' from 'units.ProduceResults_1.hidden_layers.0.weight')
                parts = key.split('.')
                if len(parts) >= 2:
                    unit_keys.add(parts[1])

        # Create each unit by parsing the key and getting exact dimensions from checkpoint
        for unit_key in unit_keys:
            # Parse operator_type and num_children from key (e.g., 'ProduceResults_1')
            parts = unit_key.rsplit('_', 1)
            if len(parts) == 2:
                operator_type = parts[0]
                num_children = int(parts[1])

                # Get the exact input dimension from the checkpoint
                # The first hidden layer weight has shape [hidden_dim, input_dim]
                weight_key = f'units.{unit_key}.hidden_layers.0.weight'
                if weight_key in state_dict:
                    weight_shape = state_dict[weight_key].shape
                    input_dim = weight_shape[1]  # input_dim is the second dimension

                    # For operators with children, we need to figure out the operator feature dim
                    # input_dim = operator_feature_dim + (data_vec_dim * num_children)
                    data_vec_dim = 32
                    operator_feature_dim = input_dim - (data_vec_dim * num_children)

                    # Directly create the unit with the correct dimensions
                    # We bypass _ensure_unit_exists to create with exact dimensions
                    from neural_units import create_neural_unit
                    unit = create_neural_unit(
                        operator_type=operator_type,
                        input_dim=operator_feature_dim,
                        num_children=num_children,
                        data_vec_dim=data_vec_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers
                    )
                    # Add to model's units
                    key = self.model._get_unit_key(operator_type, num_children)
                    self.model.units[key] = unit
                    self.model.operator_types.add(operator_type)

        print('Loading trained weights...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f'  Trained for {checkpoint["epoch"]} epochs')
        print(f'  Model has {self.model.count_parameters():,} parameters')
        print(f'  Number of neural units: {len(self.model.units)}')

    def close(self):
        """Close database connection."""
        self.extractor.close()

    def predict_latency(self, query: str) -> tuple[float, dict]:
        """
        Predict query latency using EXPLAIN

        Args:
            query: Cypher query string

        Returns:
            Tuple of (predicted_latency, plan)
        """
        # Get plan with EXPLAIN
        plan = self.extractor.get_plan(query)

        with torch.no_grad():
            predicted_latency = self.model.forward(plan)

        return predicted_latency.item(), plan

    def measure_actual_latency(self, query: str, num_runs: int = 3) -> tuple[float, float]:
        """
        Measure actual query execution time.

        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging

        Returns:
            Tuple of (mean_latency, std_latency)
        """
        execution_times = []

        with self.extractor.neo4j.session() as session:
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(cypher(query))
                result.consume()
                end_time = time.time()
                execution_times.append(end_time - start_time)

        return np.mean(execution_times).item(), np.std(execution_times).item()

    def evaluate_query(self, query: str, query_name: str | None = None, num_runs: int = 3) -> dict:
        """
        Evaluate a single query.

        Args:
            query: Cypher query string
            query_name: Optional name for the query
            num_runs: Number of executions for averaging

        Returns:
            Dictionary with evaluation results
        """
        if query_name:
            print(f'\nEvaluating: {query_name}')

        pred_latency, plan = self.predict_latency(query)

        actual_latency, std_latency = self.measure_actual_latency(query, num_runs)

        # Compute metrics
        abs_error = abs(pred_latency - actual_latency)
        r_value = max(pred_latency / actual_latency, actual_latency / pred_latency) \
                  if pred_latency > 0 and actual_latency > 0 else float('inf')

        result = {
            'query_name': query_name,
            'query': query,
            'predicted_latency': pred_latency,
            'actual_latency': actual_latency,
            'std_latency': std_latency,
            'absolute_error': abs_error,
            'r_value': r_value,
            'plan': plan
        }

        print(f'  Predicted: {pred_latency * 1000:.2f}ms')
        print(f'  Actual: {actual_latency * 1000:.2f}ms (±{std_latency * 1000:.2f}ms)')
        print(f'  Absolute Error: {abs_error * 1000:.2f}ms')
        print(f'  R-value: {r_value:.4f}')

        return result

    def evaluate_multiple_queries(self, queries: list[tuple[str, str]], num_runs: int = 3) -> list[dict]:
        """
        Evaluate multiple queries.

        Args:
            queries: List of (query_name, query) tuples
            num_runs: Number of executions per query for averaging

        Returns:
            List of evaluation result dictionaries
        """
        results = []

        print(f'\nEvaluating {len(queries)} queries...')
        print('=' * 70)

        for query_name, query in queries:
            try:
                result = self.evaluate_query(query, query_name, num_runs)
                results.append(result)
            except Exception as e:
                print(f'  ✗ Error evaluating {query_name}: {str(e)}')
                continue

        return results

    def print_summary(self, results: list[dict]):
        """Print summary statistics of evaluation results."""
        if not results:
            print('\nNo results to summarize.')
            return

        print('\n' + '=' * 70)
        print('Evaluation Summary')
        print('=' * 70)

        # Extract metrics
        abs_errors = [r['absolute_error'] for r in results]
        r_values = [r['r_value'] for r in results if r['r_value'] != float('inf')]

        # Compute statistics
        print(f'\nNumber of queries: {len(results)}')
        print(f'\nAbsolute Error:')
        print(f'  Mean: {np.mean(abs_errors) * 1000:.2f}ms')
        print(f'  Median: {np.median(abs_errors) * 1000:.2f}ms')
        print(f'  Std: {np.std(abs_errors) * 1000:.2f}ms')
        print(f'  Min/Max: {np.min(abs_errors) * 1000:.2f}ms / {np.max(abs_errors) * 1000:.2f}ms')


        if r_values:
            print(f'\nR-value:')
            print(f'  Mean: {np.mean(r_values):.4f}')
            print(f'  Median: {np.median(r_values):.4f}')
            print(f'  90th percentile: {np.percentile(r_values, 90):.4f}')
            print(f'  95th percentile: {np.percentile(r_values, 95):.4f}')
            print(f'  Min/Max: {np.min(r_values):.4f} / {np.max(r_values):.4f}')

        # Create results table
        table_data = []
        for r in results:
            table_data.append([
                r['query_name'][:30] if r['query_name'] else 'N/A',
                f'{r["predicted_latency"] * 1000:.2f}',
                f'{r["actual_latency"] * 1000:.2f}',
                f'{r["absolute_error"] * 1000:.2f}',
                f'{r["r_value"]:.4f}' if r['r_value'] != float('inf') else 'inf'
            ])

        print('\n' + '=' * 70)
        print('Detailed Results')
        print('=' * 70)
        print(tabulate(
            table_data,
            headers=['Query', 'Predicted (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))

        r_value_thresholds = [1.5, 2.0, 3.0]
        total_queries = len(results)
        print('\nR-value Thresholds:')
        for threshold in r_value_thresholds:
            count_below = sum(1 for r in results if r['r_value'] < threshold)
            percent_below = (count_below / total_queries) * 100
            print(f'  R-value < {threshold}: {count_below} queries ({percent_below:.2f}%)')



def generate_test_queries() -> list[tuple[str, str]]:
    """
    Generate a diverse set of test queries for evaluation.
    These should be different from training queries.
    """
    queries = []

    # Q1 variants (Original)
    queries.append(('Q1-Test-1', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate <= date('1998-10-15')
        WITH li.l_returnflag AS returnflag, li.l_linestatus AS linestatus, li
        RETURN
          returnflag,
          linestatus,
          sum(li.l_quantity) AS sum_qty,
          sum(li.l_extendedprice) AS sum_base_price
        ORDER BY returnflag, linestatus
        LIMIT 5
    """))

    # Q5 variant (Original)
    queries.append(('Q5-Test-1', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
              (li)-[:IS_PRODUCT_SUPPLY]->(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier),
              (c)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
        WHERE r.r_name = 'ASIA'
          AND o.o_orderdate >= date('1994-01-01')
          AND o.o_orderdate < date('1995-01-01')
          AND s.s_nationkey = n.n_nationkey
        WITH n.n_name AS nation, li
        RETURN nation, sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
        ORDER BY revenue DESC
        LIMIT 3
    """))

    # Q6 variant (Original)
    queries.append(('Q6-Test-1', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate >= date('1994-01-01')
          AND li.l_shipdate < date('1995-01-01')
          AND li.l_discount >= 0.05
          AND li.l_discount <= 0.07
          AND li.l_quantity < 24
        RETURN sum(li.l_extendedprice * li.l_discount) AS revenue
    """))

    # Q10 variant (Original)
    queries.append(('Q10-Test-1', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
              (c)-[:IS_IN_NATION]->(n:Nation)
        WHERE o.o_orderdate >= date('1993-07-01')
          AND o.o_orderdate < date('1993-10-01')
          AND li.l_returnflag = 'R'
        WITH c, n, li
        RETURN
          c.c_custkey,
          c.c_name,
          sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue,
          c.c_acctbal,
          n.n_name
        ORDER BY revenue DESC
        LIMIT 5
    """))

    # Simple aggregation query (Original)
    queries.append(('Simple-Agg-1', """
        MATCH (li:LineItem)
        WHERE li.l_quantity > 30
        RETURN count(li) AS count, avg(li.l_extendedprice) AS avg_price
    """))

    # Simple scan with limit (Original)
    queries.append(('Simple-Scan-1', """
        MATCH (c:Customer)
        WHERE c.c_acctbal > 5000
        RETURN c.c_name, c.c_acctbal
        ORDER BY c.c_acctbal DESC
        LIMIT 10
    """))

    # ========================================================================
    # CATEGORY 1: Simple Aggregation Queries
    # ========================================================================

    queries.append(('Simple Agg 1: Lineitem Summary', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate <= date('1998-08-01')
        RETURN
            li.l_returnflag AS returnflag,
            count(li) AS count,
            avg(li.l_quantity) AS avg_qty,
            sum(li.l_extendedprice) AS total_price
        ORDER BY returnflag
    """))

    queries.append(('Simple Agg 2: Order Statistics', """
        MATCH (o:Order)
        WHERE o.o_orderdate >= date('1996-01-01')
        RETURN
            o.o_orderpriority AS orderpriority,
            count(o) AS order_count,
            avg(o.o_totalprice) AS avg_price,
            min(o.o_totalprice) AS min_price,
            max(o.o_totalprice) AS max_price
        ORDER BY orderpriority
    """))

    queries.append(('Simple Agg 3: Customer Segments', """
        MATCH (c:Customer)
        WHERE c.c_acctbal > 0
        WITH
            c.c_mktsegment AS mktsegment,
            count(c) AS customer_count,
            avg(c.c_acctbal) AS avg_balance,
            sum(c.c_acctbal) AS total_balance
        RETURN
            mktsegment,
            customer_count,
            avg_balance,
            total_balance
        ORDER BY customer_count DESC
    """))

    queries.append(('Simple Agg 4: Part Analysis', """
        MATCH (p:Part)
        WHERE p.p_size >= 10 AND p.p_size <= 30
        WITH
            p.p_brand AS brand,
            p.p_type AS type,
            count(p) AS part_count,
            avg(p.p_retailprice) AS avg_price
        WHERE part_count > 5
        RETURN
            brand,
            type,
            part_count,
            avg_price
        ORDER BY brand, type
    """))

    queries.append(('Simple Agg 5: Supplier Stats', """
        MATCH (s:Supplier)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
        WHERE s.s_acctbal > 1000
        WITH
            r.r_name AS region_name,
            n.n_name AS nation_name,
            count(s) AS supplier_count,
            avg(s.s_acctbal) AS avg_balance,
            sum(s.s_acctbal) AS total_balance
        RETURN
            region_name,
            nation_name,
            supplier_count,
            avg_balance,
            total_balance
        ORDER BY supplier_count DESC
    """))

    queries.append(('Simple Agg 6: Discount Analysis', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate >= date('1997-01-01') AND li.l_shipdate < date('1998-01-01')
        RETURN
            li.l_linestatus AS linestatus,
            avg(li.l_discount) AS avg_discount,
            avg(li.l_tax) AS avg_tax,
            count(li) AS line_count
        ORDER BY linestatus
    """))

    # ========================================================================
    # CATEGORY 2: Simple Join Queries
    # ========================================================================

    queries.append(('Join 1: Customer Orders', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)
        WHERE o.o_orderdate >= date('1995-01-01')
        WITH
            c.c_name AS name,
            c.c_mktsegment AS mktsegment,
            count(o) AS order_count,
            sum(o.o_totalprice) AS total_spent
        RETURN
            name,
            mktsegment,
            order_count,
            total_spent
        ORDER BY total_spent DESC
        LIMIT 100
    """))

    queries.append(('Join 2: Parts and Suppliers', """
        MATCH (ps:PartSupp)-[:IS_FOR_PART]->(p:Part)
        WHERE p.p_size > 20 AND ps.ps_supplycost < 100
        RETURN
            p.p_partkey AS partkey,
            p.p_name AS name,
            ps.ps_supplycost AS supplycost,
            ps.ps_availqty AS availqty
        ORDER BY supplycost
        LIMIT 200
    """))

    queries.append(('Join 3: Order Details', """
        MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
        WHERE o.o_orderdate >= date('1996-01-01') AND o.o_orderdate <= date('1996-03-31')
          AND li.l_quantity > 30
        RETURN
            o.o_orderkey AS orderkey,
            o.o_orderdate AS orderdate,
            li.l_linenumber AS linenumber,
            li.l_quantity AS quantity,
            li.l_extendedprice AS extendedprice
        ORDER BY orderdate, orderkey
        LIMIT 500
    """))

    queries.append(('Join 4: Supplier Orders', """
        MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE li.l_shipdate >= date('1996-01-01') AND li.l_shipdate < date('1997-01-01')
        WITH
            s.s_name AS name,
            s.s_address AS address,
            count(DISTINCT o) AS order_count,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
        RETURN
            name,
            address,
            order_count,
            revenue
        ORDER BY revenue DESC
        LIMIT 50
    """))

    queries.append(('Join 5: Customer Nation Analysis', """
        MATCH (r:Region)<-[:IS_IN_REGION]-(n:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)
        WHERE o.o_orderdate >= date('1997-01-01')
        WITH
            r.r_name AS region_name,
            n.n_name AS nation_name,
            count(DISTINCT c) AS customer_count,
            count(o) AS total_orders,
            avg(o.o_totalprice) AS avg_order_value
        RETURN
            region_name,
            nation_name,
            customer_count,
            total_orders,
            avg_order_value
        ORDER BY total_orders DESC
    """))

    queries.append(('Join 6: Part Lineitem Summary', """
        MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li:LineItem)
        WHERE li.l_shipdate >= date('1995-01-01') AND p.p_size < 15
        WITH
            p.p_brand AS brand,
            p.p_container AS container,
            count(li) AS shipment_count,
            avg(li.l_quantity) AS avg_quantity,
            sum(li.l_extendedprice) AS total_value
        WHERE shipment_count > 10
        RETURN
            brand,
            container,
            shipment_count,
            avg_quantity,
            total_value
        ORDER BY brand, container
    """))

    # ========================================================================
    # CATEGORY 3: Complex Multi-table Joins
    # ========================================================================

    queries.append(('Complex Join 1: Customer Segment Revenue', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
        WHERE li.l_shipdate >= date('1995-06-01')
          AND li.l_shipdate < date('1995-09-01')
          AND c.c_mktsegment = 'BUILDING'
        RETURN
            c.c_mktsegment AS mktsegment,
            avg(li.l_extendedprice * (1 - li.l_discount)) AS avg_revenue,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
            count(DISTINCT c) AS customer_count
    """))

    queries.append(('Complex Join 2: Supplier Revenue Analysis', """
        MATCH (r:Region)<-[:IS_IN_REGION]-(cn:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
              (li)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)-[:IS_IN_NATION]->(sn:Nation)-[:IS_IN_REGION]->(r)
        WHERE o.o_orderdate >= date('1994-01-01')
          AND o.o_orderdate < date('1995-01-01')
        WITH
            r.r_name AS region_name,
            sn.n_name AS supplier_nation,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
        RETURN
            region_name,
            supplier_nation,
            revenue
        ORDER BY revenue DESC
        LIMIT 100
    """))

    queries.append(('Complex Join 3: Part Supplier Customer Chain', """
        MATCH (p:Part)<-[:IS_FOR_PART]-(:PartSupp)<-[:IS_PRODUCT_SUPPLY]-(li:LineItem)<-[:CONTAINS_ITEM]-(o:Order)<-[:PLACED]-(c:Customer)
        WHERE p.p_type CONTAINS 'BRASS'
          AND o.o_orderdate >= date('1996-01-01')
          AND o.o_orderdate < date('1997-01-01')
        WITH
            p.p_brand AS brand,
            c.c_mktsegment AS mktsegment,
            count(DISTINCT o) AS order_count,
            sum(li.l_quantity) AS total_quantity,
            avg(li.l_discount) AS avg_discount
        RETURN
            brand,
            mktsegment,
            order_count,
            total_quantity,
            avg_discount
        ORDER BY order_count DESC
    """))

    queries.append(('Complex Join 4: Multi-way with Partsupp', """
        MATCH (p:Part)<-[:IS_FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s:Supplier),
              (li:LineItem)-[:IS_PRODUCT_SUPPLY]->(ps)
        WHERE li.l_shipdate >= date('1996-01-01')
          AND li.l_shipdate < date('1996-06-01')
          AND p.p_size > 15
        WITH
            p.p_partkey AS partkey,
            s.s_name AS name,
            ps.ps_supplycost AS supplycost,
            sum(li.l_quantity) AS total_shipped
        WHERE total_shipped > 50
        RETURN
            partkey,
            name,
            supplycost,
            total_shipped
        ORDER BY partkey, name
        LIMIT 100
    """))

    queries.append(('Complex Join 5: Full Chain Analysis', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
        WHERE o.o_orderdate >= date('1997-01-01')
          AND o.o_orderdate < date('1997-07-01')
          AND li.l_discount > 0.05
        WITH
            c.c_mktsegment AS mktsegment,
            p.p_brand AS brand,
            count(li) AS transaction_count,
            avg(li.l_extendedprice * (1 - li.l_discount)) AS avg_net_price
        WHERE transaction_count > 5
        RETURN
            mktsegment,
            brand,
            transaction_count,
            avg_net_price
        ORDER BY mktsegment, brand
    """))

    queries.append(('Complex Join 6: Regional Supply Chain', """
        MATCH (r:Region)<-[:IS_IN_REGION]-(cn:Nation)<-[:IS_IN_NATION]-(c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS_ITEM]->(li:LineItem),
              (li)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)-[:IS_IN_NATION]->(sn:Nation)-[:IS_IN_REGION]->(r)
        WHERE o.o_orderdate >= date('1995-01-01')
          AND o.o_orderdate < date('1996-01-01')
        WITH
            sn.n_name AS supplier_nation,
            cn.n_name AS customer_nation,
            r.r_name AS region_name,
            count(DISTINCT o) AS orders,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS revenue
        WHERE orders > 10
        RETURN
            supplier_nation,
            customer_nation,
            region_name,
            orders,
            revenue
        ORDER BY revenue DESC
        LIMIT 50
    """))

    # ========================================================================
    # CATEGORY 4: Selective Scans with Filters
    # ========================================================================

    queries.append(('Selective 1: Discount Range', """
        MATCH (li:LineItem)
        WHERE li.l_discount >= 0.05 AND li.l_discount <= 0.07
          AND li.l_quantity < 24
          AND li.l_shipdate >= date('1994-01-01')
        WITH li, (li.l_extendedprice * li.l_discount) AS discount_revenue
        RETURN
            li.l_orderkey AS orderkey,
            li.l_linenumber AS linenumber,
            li.l_quantity AS quantity,
            li.l_extendedprice AS extendedprice,
            li.l_discount AS discount,
            discount_revenue
        ORDER BY discount_revenue DESC
        LIMIT 100
    """))

    queries.append(('Selective 2: High Value Orders', """
        MATCH (o:Order)
        WHERE o.o_totalprice > 300000
          AND o.o_orderdate >= date('1995-01-01')
          AND o.o_orderdate < date('1997-01-01')
        RETURN
            o.o_orderkey AS orderkey,
            o.o_custkey AS custkey,
            o.o_totalprice AS totalprice,
            o.o_orderdate AS orderdate
        ORDER BY totalprice DESC
        LIMIT 50
    """))

    queries.append(('Selective 3: Premium Customers', """
        MATCH (c:Customer)
        WHERE c.c_acctbal > 8000
          AND c.c_mktsegment IN ['AUTOMOBILE', 'MACHINERY']
        RETURN
            c.c_custkey AS custkey,
            c.c_name AS name,
            c.c_acctbal AS acctbal,
            c.c_mktsegment AS mktsegment
        ORDER BY acctbal DESC
        LIMIT 100
    """))

    queries.append(('Selective 4: Specific Part Types', """
        MATCH (p:Part)
        WHERE p.p_brand = 'Brand#23'
          AND p.p_container IN ['SM BOX', 'SM PACK']
          AND p.p_size >= 5 AND p.p_size <= 15
        RETURN
            p.p_partkey AS partkey,
            p.p_name AS name,
            p.p_brand AS brand,
            p.p_retailprice AS retailprice
        ORDER BY retailprice DESC
    """))

    queries.append(('Selective 5: Late Shipments', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate > li.l_commitdate
          AND li.l_receiptdate >= date('1996-01-01')
          AND li.l_receiptdate < date('1996-06-01')
        RETURN
            li.l_orderkey AS orderkey,
            li.l_linenumber AS linenumber,
            li.l_shipdate AS shipdate,
            li.l_commitdate AS commitdate,
            li.l_receiptdate AS receiptdate
        ORDER BY shipdate
        LIMIT 200
    """))

    queries.append(('Selective 6: Low Supply Cost', """
        MATCH (ps:PartSupp)
        WHERE ps.ps_supplycost < 50
          AND ps.ps_availqty > 5000
        RETURN
            ps.ps_partkey AS partkey,
            ps.ps_suppkey AS suppkey,
            ps.ps_supplycost AS supplycost,
            ps.ps_availqty AS availqty
        ORDER BY supplycost, availqty DESC
        LIMIT 150
    """))

    # ========================================================================
    # CATEGORY 5: Large Scans with Sorting
    # ========================================================================

    queries.append(('Large Scan 1: Sorted Lineitem by Price', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate >= date('1997-01-01')
          AND li.l_shipdate < date('1997-04-01')
        RETURN
            li.l_orderkey AS orderkey,
            li.l_partkey AS partkey,
            li.l_suppkey AS suppkey,
            li.l_quantity AS quantity,
            li.l_extendedprice AS extendedprice
        ORDER BY extendedprice DESC
        LIMIT 200
    """))

    queries.append(('Large Scan 2: Orders by Date', """
        MATCH (o:Order)
        WHERE o.o_orderdate >= date('1996-01-01')
          AND o.o_orderdate < date('1997-01-01')
        RETURN
            o.o_orderkey AS orderkey,
            o.o_custkey AS custkey,
            o.o_totalprice AS totalprice,
            o.o_orderdate AS orderdate,
            o.o_orderpriority AS orderpriority
        ORDER BY orderdate DESC, totalprice DESC
        LIMIT 500
    """))

    queries.append(('Large Scan 3: Parts by Price', """
        MATCH (p:Part)
        WHERE p.p_retailprice > 1000
        RETURN
            p.p_partkey AS partkey,
            p.p_name AS name,
            p.p_brand AS brand,
            p.p_type AS type,
            p.p_retailprice AS retailprice
        ORDER BY retailprice DESC, partkey
        LIMIT 300
    """))

    queries.append(('Large Scan 4: Customer Balance Ranking', """
        MATCH (c:Customer)-[:IS_IN_NATION]->(n:Nation)-[:IS_IN_REGION]->(r:Region)
        WHERE c.c_acctbal > 0
        RETURN
            c.c_custkey AS custkey,
            c.c_name AS name,
            c.c_acctbal AS acctbal,
            c.c_mktsegment AS mktsegment,
            n.n_name AS nation_name,
            r.r_name AS region_name
        ORDER BY acctbal DESC, name
        LIMIT 400
    """))

    queries.append(('Large Scan 5: Lineitem Quantity Sort', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate >= date('1996-06-01')
          AND li.l_shipdate < date('1996-09-01')
          AND li.l_quantity > 40
        RETURN
            li.l_orderkey AS orderkey,
            li.l_partkey AS partkey,
            li.l_quantity AS quantity,
            li.l_extendedprice AS extendedprice,
            li.l_discount AS discount
        ORDER BY quantity DESC, extendedprice DESC
        LIMIT 250
    """))

    queries.append(('Large Scan 6: Recent Shipments', """
        MATCH (li:LineItem)
        WHERE li.l_shipdate >= date('1998-06-01')
        WITH li, (li.l_extendedprice * (1 - li.l_discount)) as net_price
        RETURN
            li.l_orderkey AS orderkey,
            li.l_linenumber AS linenumber,
            li.l_shipdate AS shipdate,
            li.l_receiptdate AS receiptdate,
            net_price
        ORDER BY shipdate DESC, net_price DESC
        LIMIT 300
    """))

    # ========================================================================
    # CATEGORY 6: Aggregation with HAVING
    # ========================================================================

    queries.append(('Having 1: Large Order Aggregates', """
        MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)
        WITH
            o.o_orderkey AS orderkey,
            sum(li.l_quantity) AS total_qty,
            sum(li.l_extendedprice) AS total_price,
            count(li) AS line_count
        WHERE total_qty > 300
        RETURN
            orderkey,
            total_qty,
            total_price,
            line_count
        ORDER BY total_price DESC
        LIMIT 100
    """))

    queries.append(('Having 2: High Volume Customers', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)
        WHERE o.o_orderdate >= date('1996-01-01')
        WITH
            c.c_custkey AS custkey,
            count(o) AS order_count,
            sum(o.o_totalprice) AS total_spent,
            avg(o.o_totalprice) AS avg_order_value
        WHERE order_count > 15
        RETURN
            custkey,
            order_count,
            total_spent,
            avg_order_value
        ORDER BY total_spent DESC
        LIMIT 50
    """))

    queries.append(('Having 3: Popular Parts by Brand', """
        MATCH (p:Part)
        WITH
            p.p_brand AS brand,
            count(p) AS part_count,
            avg(p.p_retailprice) AS avg_price,
            min(p.p_retailprice) AS min_price,
            max(p.p_retailprice) AS max_price
        WHERE part_count > 50
        RETURN
            brand,
            part_count,
            avg_price,
            min_price,
            max_price
        ORDER BY avg_price DESC
    """))

    queries.append(('Having 4: High Revenue Suppliers', """
        MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:SUPPLIED_BY]->(s:Supplier)
        WHERE li.l_shipdate >= date('1997-01-01')
        WITH
            s.s_suppkey AS suppkey,
            count(DISTINCT o) AS order_count,
            sum(li.l_extendedprice * (1 - li.l_discount)) AS total_revenue,
            avg(li.l_quantity) AS avg_quantity
        WHERE total_revenue > 500000
        RETURN
            suppkey,
            order_count,
            total_revenue,
            avg_quantity
        ORDER BY total_revenue DESC
        LIMIT 75
    """))

    queries.append(('Having 5: Part Categories with High Sales', """
        MATCH (o:Order)-[:CONTAINS_ITEM]->(li:LineItem)-[:IS_PRODUCT_SUPPLY]->(:PartSupp)-[:IS_FOR_PART]->(p:Part)
        WHERE li.l_shipdate >= date('1996-01-01')
          AND li.l_shipdate < date('1997-01-01')
        WITH
            p.p_type AS type,
            count(DISTINCT o) AS order_count,
            sum(li.l_quantity) AS total_quantity,
            avg(li.l_extendedprice) AS avg_price
        WHERE total_quantity > 1000
        RETURN
            type,
            order_count,
            total_quantity,
            avg_price
        ORDER BY total_quantity DESC
        LIMIT 25
    """))

    queries.append(('Having 6: Customer Segments with Volume', """
        MATCH (c:Customer)-[:PLACED]->(o:Order)
        WHERE o.o_orderdate >= date('1997-01-01')
        WITH
            c.c_mktsegment AS mktsegment,
            count(DISTINCT c) AS customer_count,
            count(o) AS total_orders,
            avg(o.o_totalprice) AS avg_order_price
        WHERE total_orders > 1000
        RETURN
            mktsegment,
            customer_count,
            total_orders,
            avg_order_price
        ORDER BY total_orders DESC
    """))


    return queries

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Neo4j QPP model')
    parser.add_argument('--checkpoint', type=str, default='data/neo4j_qpp_checkpoint_best.pt', help='Path to model checkpoint')
    parser.add_argument('--feature-extractor', type=str, default='data/neo4j_qpp_checkpoint_feature_extractor.pkl', help='Path to feature extractor')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of executions per query for averaging')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--num-layers', type=int, default=10, help='Number of layers in the neural units')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size in the neural units')
    parser.add_argument('--query', '-q', type=str, action='append', dest='queries', help='Additional Cypher query to evaluate (can be used multiple times)')
    parser.add_argument('--query-only', '-qo', action='store_true', help='Only evaluate the provided --query arguments, skip built-in test queries')

    args = parser.parse_args()

    print('=' * 70)
    print('Neo4j Query Performance Predictor - Evaluation')
    print('=' * 70)

    # Load model
    config = Config.load()
    neo4j = Neo4j(config.neo4j)
    evaluator = ModelEvaluator(
        neo4j=neo4j,
        checkpoint_path=args.checkpoint,
        feature_extractor_path=args.feature_extractor,
        device=args.device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim
    )

    try:
        # Generate test queries based on arguments
        print('\nGenerating test queries...')
        if args.query_only:
            test_queries = []
        else:
            test_queries = generate_test_queries()

        # Add user-provided queries
        if args.queries:
            for i, query in enumerate(args.queries, 1):
                query_name = f'Custom Query {i}'
                test_queries.append((query_name, query))
            print(f'Added {len(args.queries)} custom query/queries')

        if not test_queries:
            print('Error: No queries to evaluate. Provide queries with --query or remove --query-only flag.')
            return

        print(f'Total queries to evaluate: {len(test_queries)}')

        results = evaluator.evaluate_multiple_queries(test_queries, args.num_runs)
        evaluator.print_summary(results)


    finally:
        evaluator.close()

if __name__ == '__main__':
    main()
