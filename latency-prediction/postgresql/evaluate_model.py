import torch
import numpy as np
import time
import json
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
from config import DatabaseConfig
from plan_extractor import PlanExtractor
from plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained QPP-Net model on new queries."""

    def __init__(self, model_path: str = 'qpp_net_model.pt'):
        """
        Load trained model and initialize evaluator.

        Args:
            model_path: Path to saved model checkpoint
        """
        print('Loading trained model...')
        checkpoint = torch.load(model_path, weights_only=False)

        self.feature_extractor = checkpoint['feature_extractor']
        config = checkpoint['config']

        # Recreate model architecture
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            data_vec_dim=config['data_vec_dim']
        )

        # Initialize units from saved operator info
        if 'operator_info' in checkpoint:
            self.model.initialize_units_from_operator_info(checkpoint['operator_info'])
        else:
            raise ValueError('Model checkpoint does not contain operator_info. Please retrain the model with the updated main.py script.')

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Print training metrics if available
        if 'metrics' in checkpoint:
            print('\nTraining Metrics:')
            print(f'  Train MAE: {checkpoint['metrics']['train']['mae']:.2f} ms')
            print(f'  Test MAE: {checkpoint['metrics']['test']['mae']:.2f} ms')
            print(f'  Test R ≤ 1.5: {checkpoint['metrics']['test']['r_within_1.5']*100:.1f}%')

        print('Model loaded successfully!\n')

    def get_query_plan_only(self, query: str) -> dict:
        """
        Get query plan WITHOUT executing the query.
        Uses EXPLAIN (without ANALYZE).

        Args:
            query: SQL query string

        Returns:
            Query plan dictionary
        """
        config = DatabaseConfig()
        conn = config.get_connection()
        conn.autocommit = True

        try:
            with conn.cursor() as cursor:
                explain_query = f'EXPLAIN (FORMAT JSON, VERBOSE) {query}'
                cursor.execute(explain_query)
                result = cursor.fetchone()
                plan = result[0][0]['Plan']
                return plan
        finally:
            conn.close()

    def get_explain_analyze_time(self, query: str) -> tuple[dict, float]:
        """
        Get query plan and execution time from EXPLAIN ANALYZE.

        Args:
            query: SQL query string

        Returns:
            Tuple of (plan, explain_analyze_time_ms)
        """
        config = DatabaseConfig()
        conn = config.get_connection()
        conn.autocommit = True

        try:
            with conn.cursor() as cursor:
                explain_query = f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}'
                cursor.execute(explain_query)
                result = cursor.fetchone()
                plan_json = result[0][0]
                execution_time = plan_json['Execution Time']
                return plan_json['Plan'], execution_time
        finally:
            conn.close()

    def get_actual_execution_time(self, query: str, num_runs: int = 3) -> tuple[float, float, float]:
        """
        Execute query and measure actual wall-clock time.
        Runs multiple times and returns statistics.

        Args:
            query: SQL query string
            num_runs: Number of times to execute the query

        Returns:
            Tuple of (mean_time_ms, min_time_ms, max_time_ms)
        """
        config = DatabaseConfig()
        times = []

        for i in range(num_runs):
            conn = config.get_connection()
            conn.autocommit = True

            try:
                with conn.cursor() as cursor:
                    start_time = time.time()
                    cursor.execute(query)
                    # Fetch all results to ensure query completes
                    cursor.fetchall()
                    end_time = time.time()

                    elapsed_ms = (end_time - start_time) * 1000
                    times.append(elapsed_ms)
            finally:
                conn.close()

        return np.mean(times), np.min(times), np.max(times)

    def predict_latency(self, plan: dict) -> float:
        """
        Predict query latency using the trained model.

        Args:
            plan: Query plan dictionary

        Returns:
            Predicted latency in milliseconds
        """
        with torch.no_grad():
            prediction = self.model(plan).item()
        return prediction

    def evaluate_query(self, query: str, query_name: str = None, measure_actual: bool = True, num_runs: int = 3) -> dict:
        """
        Comprehensive evaluation of a single query.

        Args:
            query: SQL query string
            query_name: Optional name for the query
            measure_actual: Whether to measure actual execution time
            num_runs: Number of runs for actual execution measurement

        Returns:
            Dictionary with all measurements and comparisons
        """
        if query_name is None:
            query_name = f'Query_{hash(query) % 10000}'

        print(f'\nEvaluating: {query_name}')
        print('-' * 80)

        results = {'query_name': query_name, 'query': query}

        # 1. Get prediction (without executing)
        print('  [1/3] Getting query plan and prediction...')
        plan = self.get_query_plan_only(query)
        predicted_time = self.predict_latency(plan)
        results['predicted_time'] = predicted_time
        print(f'        Model Prediction: {predicted_time:.2f} ms')

        # 2. Get EXPLAIN ANALYZE time
        print('  [2/3] Running EXPLAIN ANALYZE...')
        _, explain_time = self.get_explain_analyze_time(query)
        results['explain_analyze_time'] = explain_time
        print(f'        EXPLAIN ANALYZE: {explain_time:.2f} ms')

        # 3. Get actual execution time
        if measure_actual:
            print(f'  [3/3] Measuring actual execution ({num_runs} runs)...')
            actual_mean, actual_min, actual_max = self.get_actual_execution_time(query, num_runs)
            results['actual_time_mean'] = actual_mean
            results['actual_time_min'] = actual_min
            results['actual_time_max'] = actual_max
            print(f'        Actual Time: {actual_mean:.2f} ms (min: {actual_min:.2f}, max: {actual_max:.2f})')
        else:
            results['actual_time_mean'] = None
            results['actual_time_min'] = None
            results['actual_time_max'] = None

        # Calculate errors and ratios
        results['error_vs_explain'] = abs(predicted_time - explain_time)
        results['relative_error_vs_explain'] = abs(predicted_time - explain_time) / (explain_time + 1e-8)
        results['r_vs_explain'] = max(predicted_time / (explain_time + 1e-8),
                                       explain_time / (predicted_time + 1e-8))
        results['explain_predicted_ratio'] = explain_time / (predicted_time + 1e-8)

        if measure_actual:
            results['error_vs_actual'] = abs(predicted_time - actual_mean)
            results['relative_error_vs_actual'] = abs(predicted_time - actual_mean) / (actual_mean + 1e-8)
            results['r_vs_actual'] = max(predicted_time / (actual_mean + 1e-8),
                                          actual_mean / (predicted_time + 1e-8))
            results['actual_predicted_ratio'] = actual_mean / (predicted_time + 1e-8)

            # Also compare EXPLAIN ANALYZE vs Actual
            results['explain_vs_actual_error'] = abs(explain_time - actual_mean)
            results['explain_vs_actual_relative'] = abs(explain_time - actual_mean) / (actual_mean + 1e-8)

        print(f'\n  Prediction Error vs EXPLAIN ANALYZE: {results['error_vs_explain']:.2f} ms '
              f'(R={results['r_vs_explain']:.2f})')

        if measure_actual:
            print(f'  Prediction Error vs Actual: {results['error_vs_actual']:.2f} ms '
                  f'(R={results['r_vs_actual']:.2f})')
            print(f'  EXPLAIN ANALYZE vs Actual: {results['explain_vs_actual_error']:.2f} ms')

        return results

    def evaluate_multiple_queries(self, queries: list[tuple[str, str]], measure_actual: bool = True, num_runs: int = 3) -> list[dict]:
        """
        Evaluate multiple queries.

        Args:
            queries: List of (query_name, query_sql) tuples
            measure_actual: Whether to measure actual execution times
            num_runs: Number of runs for each query

        Returns:
            List of result dictionaries
        """
        results = []

        print('=' * 80)
        print(f'EVALUATING {len(queries)} QUERIES')
        print('=' * 80)

        for query_name, query in queries:
            try:
                result = self.evaluate_query(query, query_name, measure_actual, num_runs)
                results.append(result)
            except Exception as e:
                print(f'\nError evaluating {query_name}: {e}')
                import traceback
                traceback.print_exc()
                continue

        return results

    def print_summary(self, results: list[dict]):
        """Print summary statistics and comparison table."""

        print('\n' + '=' * 80)
        print('EVALUATION SUMMARY')
        print('=' * 80)

        # Prepare data for table
        table_data = []
        for r in results:
            row = [
                r['query_name'],
                f'{r['predicted_time']:.1f}',
                f'{r['explain_analyze_time']:.1f}',
                f'{r['actual_time_mean']:.1f}' if r['actual_time_mean'] else 'N/A',
                f'{r['explain_predicted_ratio']:.2f}',
                f'{r['r_vs_explain']:.2f}',
            ]
            if r['actual_time_mean']:
                row.extend([
                    f'{r['actual_predicted_ratio']:.2f}',
                    f'{r['r_vs_actual']:.2f}',
                ])
            table_data.append(row)

        headers = [
            'Query',
            'Predicted\n(ms)',
            'EXPLAIN\n(ms)',
            'Actual\n(ms)',
            'Explain/\nPredicted',
            'R vs\nEXPLAIN'
        ]

        if results[0]['actual_time_mean']:
            headers.extend(['Actual/\nPredicted', 'R vs\nActual'])

        print('\n' + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Summary statistics
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        # Prediction vs EXPLAIN ANALYZE
        errors_explain = [r['error_vs_explain'] for r in results]
        relative_errors_explain = [r['relative_error_vs_explain'] for r in results]
        r_values_explain = [r['r_vs_explain'] for r in results]

        print('\nModel Prediction vs EXPLAIN ANALYZE:')
        print(f'  Mean Absolute Error: {np.mean(errors_explain):.2f} ms')
        print(f'  Median Absolute Error: {np.median(errors_explain):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(relative_errors_explain):.4f}')
        print(f'  Median R-value: {np.median(r_values_explain):.2f}')
        print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_explain])*100:.1f}%')
        print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_explain])*100:.1f}%')

        # Prediction vs Actual
        if results[0]['actual_time_mean']:
            errors_actual = [r['error_vs_actual'] for r in results]
            relative_errors_actual = [r['relative_error_vs_actual'] for r in results]
            r_values_actual = [r['r_vs_actual'] for r in results]

            print('\nModel Prediction vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(errors_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(errors_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(relative_errors_actual):.4f}')
            print(f'  Median R-value: {np.median(r_values_actual):.2f}')
            print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_actual])*100:.1f}%')
            print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_actual])*100:.1f}%')

            # EXPLAIN ANALYZE vs Actual
            explain_vs_actual = [r['explain_vs_actual_error'] for r in results]
            explain_vs_actual_rel = [r['explain_vs_actual_relative'] for r in results]

            print('\nEXPLAIN ANALYZE vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(explain_vs_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(explain_vs_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(explain_vs_actual_rel):.4f}')

    def plot_results(self, results: list[dict], save_path: str = 'evaluation_plots.png'):
        """Create visualization plots comparing predictions and actual times."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QPP-Net Model Evaluation', fontsize=16, fontweight='bold')

        predicted = [r['predicted_time'] for r in results]
        explain = [r['explain_analyze_time'] for r in results]

        # Plot 1: Predicted vs EXPLAIN ANALYZE
        ax1 = axes[0, 0]
        ax1.scatter(explain, predicted, alpha=0.6)
        min_val = min(min(explain), min(predicted))
        max_val = max(max(explain), max(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax1.set_xlabel('EXPLAIN ANALYZE Time (ms)')
        ax1.set_ylabel('Predicted Time (ms)')
        ax1.set_title('Prediction vs EXPLAIN ANALYZE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error distribution (Predicted vs EXPLAIN)
        ax2 = axes[0, 1]
        errors = [r['error_vs_explain'] for r in results]
        ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Absolute Error (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution (vs EXPLAIN ANALYZE)')
        ax2.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f}ms')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        if results[0]['actual_time_mean']:
            actual = [r['actual_time_mean'] for r in results]

            # Plot 3: Predicted vs Actual
            ax3 = axes[1, 0]
            ax3.scatter(actual, predicted, alpha=0.6, color='green')
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax3.set_xlabel('Actual Execution Time (ms)')
            ax3.set_ylabel('Predicted Time (ms)')
            ax3.set_title('Prediction vs Actual Execution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: EXPLAIN ANALYZE vs Actual
            ax4 = axes[1, 1]
            ax4.scatter(actual, explain, alpha=0.6, color='orange')
            min_val = min(min(actual), min(explain))
            max_val = max(max(actual), max(explain))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
            ax4.set_xlabel('Actual Execution Time (ms)')
            ax4.set_ylabel('EXPLAIN ANALYZE Time (ms)')
            ax4.set_title('EXPLAIN ANALYZE vs Actual Execution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Plot 3: R-value distribution
            ax3 = axes[1, 0]
            r_values = [r['r_vs_explain'] for r in results]
            ax3.hist(r_values, bins=20, edgecolor='black', alpha=0.7)
            ax3.set_xlabel('R-value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('R-value Distribution (vs EXPLAIN ANALYZE)')
            ax3.axvline(1.5, color='r', linestyle='--', label='R=1.5 threshold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            # Hide plot 4
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\nPlots saved to: {save_path}')

        return fig


def generate_test_queries() -> list[tuple[str, str]]:
    queries = []

    # ========================================================================
    # CATEGORY 1: Simple Aggregation Queries
    # ========================================================================

    queries.append(('Simple Agg 1: Lineitem Summary', """
        SELECT
            l_returnflag,
            COUNT(*) as count,
            AVG(l_quantity) as avg_qty,
            SUM(l_extendedprice) as total_price
        FROM lineitem
        WHERE l_shipdate <= date '1998-08-01'
        GROUP BY l_returnflag
    """))

    queries.append(('Simple Agg 2: Order Statistics', """
        SELECT
            o_orderpriority,
            COUNT(*) as order_count,
            AVG(o_totalprice) as avg_price,
            MIN(o_totalprice) as min_price,
            MAX(o_totalprice) as max_price
        FROM orders
        WHERE o_orderdate >= date '1996-01-01'
        GROUP BY o_orderpriority
    """))

    queries.append(('Simple Agg 3: Customer Segments', """
        SELECT
            c_mktsegment,
            COUNT(*) as customer_count,
            AVG(c_acctbal) as avg_balance,
            SUM(c_acctbal) as total_balance
        FROM customer
        WHERE c_acctbal > 0
        GROUP BY c_mktsegment
        ORDER BY customer_count DESC
    """))

    queries.append(('Simple Agg 4: Part Analysis', """
        SELECT
            p_brand,
            p_type,
            COUNT(*) as part_count,
            AVG(p_retailprice) as avg_price
        FROM part
        WHERE p_size BETWEEN 10 AND 30
        GROUP BY p_brand, p_type
        HAVING COUNT(*) > 5
    """))

    queries.append(('Simple Agg 5: Supplier Stats', """
        SELECT
            r.r_name AS region_name,
            n.n_name AS nation_name,
            COUNT(*) AS supplier_count,
            AVG(s.s_acctbal) AS avg_balance,
            SUM(s.s_acctbal) AS total_balance
        FROM supplier AS s
        JOIN nation AS n ON s.s_nationkey = n.n_nationkey
        JOIN region AS r ON n.n_regionkey = r.r_regionkey
        WHERE s.s_acctbal > 1000
        GROUP BY r.r_name, n.n_name
        ORDER BY supplier_count DESC
    """))

    queries.append(('Simple Agg 6: Discount Analysis', """
        SELECT
            l_linestatus,
            AVG(l_discount) as avg_discount,
            AVG(l_tax) as avg_tax,
            COUNT(*) as line_count
        FROM lineitem
        WHERE l_shipdate >= date '1997-01-01'
            AND l_shipdate < date '1998-01-01'
        GROUP BY l_linestatus
    """))

    # ========================================================================
    # CATEGORY 2: Simple Join Queries
    # ========================================================================

    queries.append(('Join 1: Customer Orders', """
        SELECT
            c_name,
            c_mktsegment,
            COUNT(o_orderkey) as order_count,
            SUM(o_totalprice) as total_spent
        FROM customer
        JOIN orders ON c_custkey = o_custkey
        WHERE o_orderdate >= date '1995-01-01'
        GROUP BY c_custkey, c_name, c_mktsegment
        ORDER BY total_spent DESC
        LIMIT 100
    """))

    queries.append(('Join 2: Parts and Suppliers', """
        SELECT
            p_partkey,
            p_name,
            ps_supplycost,
            ps_availqty
        FROM part
        JOIN partsupp ON p_partkey = ps_partkey
        WHERE p_size > 20
            AND ps_supplycost < 100
        ORDER BY ps_supplycost
        LIMIT 200
    """))

    queries.append(('Join 3: Order Details', """
        SELECT
            o_orderkey,
            o_orderdate,
            l_linenumber,
            l_quantity,
            l_extendedprice
        FROM orders
        JOIN lineitem ON o_orderkey = l_orderkey
        WHERE o_orderdate BETWEEN date '1996-01-01' AND date '1996-03-31'
            AND l_quantity > 30
        ORDER BY o_orderdate, o_orderkey
        LIMIT 500
    """))

    queries.append(('Join 4: Supplier Orders', """
        SELECT
            s_name,
            s_address,
            COUNT(DISTINCT l_orderkey) as order_count,
            SUM(l_extendedprice * (1 - l_discount)) as revenue
        FROM supplier
        JOIN lineitem ON s_suppkey = l_suppkey
        WHERE l_shipdate >= date '1996-01-01'
            AND l_shipdate < date '1997-01-01'
        GROUP BY s_suppkey, s_name, s_address
        ORDER BY revenue DESC
        LIMIT 50
    """))

    queries.append(('Join 5: Customer Nation Analysis', """
        SELECT
            r.r_name AS region_name,
            n.n_name AS nation_name,
            COUNT(DISTINCT c.c_custkey) AS customer_count,
            COUNT(o.o_orderkey) AS total_orders,
            AVG(o.o_totalprice) AS avg_order_value
        FROM region AS r
        JOIN nation AS n ON n.n_regionkey = r.r_regionkey
        JOIN customer AS c ON c.c_nationkey = n.n_nationkey
        JOIN orders AS o ON o.o_custkey = c.c_custkey
        WHERE o.o_orderdate >= date '1997-01-01'
        GROUP BY r.r_name, n.n_name
        ORDER BY total_orders DESC
    """))

    queries.append(('Join 6: Part Lineitem Summary', """
        SELECT
            p_brand,
            p_container,
            COUNT(*) as shipment_count,
            AVG(l_quantity) as avg_quantity,
            SUM(l_extendedprice) as total_value
        FROM part
        JOIN lineitem ON p_partkey = l_partkey
        WHERE l_shipdate >= date '1995-01-01'
            AND p_size < 15
        GROUP BY p_brand, p_container
        HAVING COUNT(*) > 10
    """))

    # ========================================================================
    # CATEGORY 3: Complex Multi-table Joins
    # ========================================================================

    queries.append(('Complex Join 1: Customer Segment Revenue', """
        SELECT
            c_mktsegment,
            AVG(l_extendedprice * (1 - l_discount)) as avg_revenue,
            SUM(l_extendedprice * (1 - l_discount)) as total_revenue,
            COUNT(DISTINCT c_custkey) as customer_count
        FROM customer
        JOIN orders ON c_custkey = o_custkey
        JOIN lineitem ON o_orderkey = l_orderkey
        WHERE l_shipdate >= date '1995-06-01'
            AND l_shipdate < date '1995-09-01'
            AND c_mktsegment = 'BUILDING'
        GROUP BY c_mktsegment
    """))

    queries.append(('Complex Join 2: Supplier Revenue Analysis', """
        SELECT
            r.r_name AS region_name,
            sn.n_name AS supplier_nation,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
        FROM region AS r
        JOIN nation AS cn ON cn.n_regionkey = r.r_regionkey
        JOIN customer AS c ON c.c_nationkey = cn.n_nationkey
        JOIN orders AS o ON o.o_custkey = c.c_custkey
        JOIN lineitem AS l ON l.l_orderkey = o.o_orderkey
        JOIN supplier AS s ON s.s_suppkey = l.l_suppkey
        JOIN nation AS sn ON sn.n_nationkey = s.s_nationkey
        WHERE sn.n_regionkey = r.r_regionkey
            AND o.o_orderdate >= date '1994-01-01'
            AND o.o_orderdate < date '1995-01-01'
        GROUP BY r.r_name, sn.n_name
        ORDER BY revenue DESC
        LIMIT 100
    """))

    queries.append(('Complex Join 3: Part Supplier Customer Chain', """
        SELECT
            p_brand,
            c_mktsegment,
            COUNT(DISTINCT o_orderkey) as order_count,
            SUM(l_quantity) as total_quantity,
            AVG(l_discount) as avg_discount
        FROM part
        JOIN lineitem ON p_partkey = l_partkey
        JOIN orders ON l_orderkey = o_orderkey
        JOIN customer ON o_custkey = c_custkey
        WHERE p_type LIKE '%BRASS%'
            AND o_orderdate >= date '1996-01-01'
            AND o_orderdate < date '1997-01-01'
        GROUP BY p_brand, c_mktsegment
        ORDER BY order_count DESC
    """))

    queries.append(('Complex Join 4: Multi-way with Partsupp', """
        SELECT
            p_partkey,
            s_name,
            ps_supplycost,
            SUM(l_quantity) as total_shipped
        FROM part
        JOIN partsupp ON p_partkey = ps_partkey
        JOIN supplier ON ps_suppkey = s_suppkey
        JOIN lineitem ON p_partkey = l_partkey AND s_suppkey = l_suppkey
        WHERE l_shipdate >= date '1996-01-01'
            AND l_shipdate < date '1996-06-01'
            AND p_size > 15
        GROUP BY p_partkey, s_suppkey, s_name, ps_supplycost
        HAVING SUM(l_quantity) > 50
        LIMIT 100
    """))

    queries.append(('Complex Join 5: Full Chain Analysis', """
        SELECT
            c_mktsegment,
            p_brand,
            COUNT(*) as transaction_count,
            AVG(l_extendedprice * (1 - l_discount)) as avg_net_price
        FROM customer
        JOIN orders ON c_custkey = o_custkey
        JOIN lineitem ON o_orderkey = l_orderkey
        JOIN part ON l_partkey = p_partkey
        WHERE o_orderdate >= date '1997-01-01'
            AND o_orderdate < date '1997-07-01'
            AND l_discount > 0.05
        GROUP BY c_mktsegment, p_brand
        HAVING COUNT(*) > 5
    """))

    queries.append(('Complex Join 6: Regional Supply Chain', """
        SELECT
            sn.n_name AS supplier_nation,
            cn.n_name AS customer_nation,
            r.r_name AS region_name,
            COUNT(DISTINCT o.o_orderkey) AS orders,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
        FROM region AS r
        JOIN nation AS cn ON cn.n_regionkey = r.r_regionkey
        JOIN customer AS c ON c.c_nationkey = cn.n_nationkey
        JOIN orders AS o ON o.o_custkey = c.c_custkey
        JOIN lineitem AS l ON l.l_orderkey = o.o_orderkey
        JOIN supplier AS s ON s.s_suppkey = l.l_suppkey
        JOIN nation AS sn ON sn.n_nationkey = s.s_nationkey
        WHERE sn.n_regionkey = r.r_regionkey
            AND o.o_orderdate >= date '1995-01-01'
            AND o.o_orderdate < date '1996-01-01'
        GROUP BY sn.n_name, cn.n_name, r.r_name
        HAVING COUNT(DISTINCT o.o_orderkey) > 10
        ORDER BY revenue DESC
        LIMIT 50
    """))

    # ========================================================================
    # CATEGORY 4: Selective Scans with Filters
    # ========================================================================

    queries.append(('Selective 1: Discount Range', """
        SELECT
            l_orderkey,
            l_linenumber,
            l_quantity,
            l_extendedprice,
            l_discount,
            l_extendedprice * l_discount as discount_revenue
        FROM lineitem
        WHERE l_discount BETWEEN 0.05 AND 0.07
            AND l_quantity < 24
            AND l_shipdate >= date '1994-01-01'
        ORDER BY discount_revenue DESC
        LIMIT 100
    """))

    queries.append(('Selective 2: High Value Orders', """
        SELECT
            o_orderkey,
            o_custkey,
            o_totalprice,
            o_orderdate
        FROM orders
        WHERE o_totalprice > 300000
            AND o_orderdate >= date '1995-01-01'
            AND o_orderdate < date '1997-01-01'
        ORDER BY o_totalprice DESC
        LIMIT 50
    """))

    queries.append(('Selective 3: Premium Customers', """
        SELECT
            c_custkey,
            c_name,
            c_acctbal,
            c_mktsegment
        FROM customer
        WHERE c_acctbal > 8000
            AND c_mktsegment IN ('AUTOMOBILE', 'MACHINERY')
        ORDER BY c_acctbal DESC
        LIMIT 100
    """))

    queries.append(('Selective 4: Specific Part Types', """
        SELECT
            p_partkey,
            p_name,
            p_brand,
            p_retailprice
        FROM part
        WHERE p_brand = 'Brand#23'
            AND p_container IN ('SM BOX', 'SM PACK')
            AND p_size BETWEEN 5 AND 15
        ORDER BY p_retailprice DESC
    """))

    queries.append(('Selective 5: Late Shipments', """
        SELECT
            l_orderkey,
            l_linenumber,
            l_shipdate,
            l_commitdate,
            l_receiptdate
        FROM lineitem
        WHERE l_shipdate > l_commitdate
            AND l_receiptdate >= date '1996-01-01'
            AND l_receiptdate < date '1996-06-01'
        ORDER BY l_shipdate
        LIMIT 200
    """))

    queries.append(('Selective 6: Low Supply Cost', """
        SELECT
            ps_partkey,
            ps_suppkey,
            ps_supplycost,
            ps_availqty
        FROM partsupp
        WHERE ps_supplycost < 50
            AND ps_availqty > 5000
        ORDER BY ps_supplycost, ps_availqty DESC
        LIMIT 150
    """))

    # ========================================================================
    # CATEGORY 5: Subquery Patterns
    # ========================================================================

    queries.append(('Subquery 1: High Value Customers', """
        SELECT
            c_custkey,
            c_name,
            c_acctbal,
            c_mktsegment
        FROM customer
        WHERE c_custkey IN (
            SELECT o_custkey
            FROM orders
            WHERE o_totalprice > 200000
                AND o_orderdate >= date '1995-01-01'
        )
        ORDER BY c_acctbal DESC
        LIMIT 50
    """))

    queries.append(('Subquery 2: Frequent Buyers', """
        SELECT
            c_custkey,
            c_name,
            c_phone
        FROM customer
        WHERE c_custkey IN (
            SELECT o_custkey
            FROM orders
            WHERE o_orderdate >= date '1997-01-01'
            GROUP BY o_custkey
            HAVING COUNT(*) > 10
        )
        ORDER BY c_name
        LIMIT 100
    """))

    queries.append(('Subquery 3: Popular Parts', """
        SELECT
            p_partkey,
            p_name,
            p_brand,
            p_retailprice
        FROM part
        WHERE p_partkey IN (
            SELECT l_partkey
            FROM lineitem
            WHERE l_shipdate >= date '1997-01-01'
                AND l_shipdate < date '1998-01-01'
            GROUP BY l_partkey
            HAVING SUM(l_quantity) > 500
        )
        ORDER BY p_retailprice DESC
    """))

    queries.append(('Subquery 4: Top Suppliers by Volume', """
        SELECT
            s_suppkey,
            s_name,
            s_address,
            s_phone
        FROM supplier
        WHERE s_suppkey IN (
            SELECT l_suppkey
            FROM lineitem
            WHERE l_shipdate >= date '1996-01-01'
            GROUP BY l_suppkey
            HAVING SUM(l_quantity) > 10000
        )
        ORDER BY s_name
    """))

    queries.append(('Subquery 5: Orders Above Average', """
        SELECT
            o_orderkey,
            o_custkey,
            o_totalprice,
            o_orderdate
        FROM orders
        WHERE o_totalprice > (
            SELECT AVG(o_totalprice)
            FROM orders
            WHERE o_orderdate >= date '1996-01-01'
        )
        AND o_orderdate >= date '1997-01-01'
        ORDER BY o_totalprice DESC
        LIMIT 100
    """))

    queries.append(('Subquery 6: Customers with Recent Large Orders', """
        SELECT
            c_custkey,
            c_name,
            c_mktsegment
        FROM customer
        WHERE EXISTS (
            SELECT 1
            FROM orders
            WHERE o_custkey = c_custkey
                AND o_totalprice > 250000
                AND o_orderdate >= date '1997-01-01'
        )
        ORDER BY c_name
        LIMIT 50
    """))

    # ========================================================================
    # CATEGORY 6: Large Scans with Sorting
    # ========================================================================

    queries.append(('Large Scan 1: Sorted Lineitem by Price', """
        SELECT
            l_orderkey,
            l_partkey,
            l_suppkey,
            l_quantity,
            l_extendedprice
        FROM lineitem
        WHERE l_shipdate >= date '1997-01-01'
            AND l_shipdate < date '1997-04-01'
        ORDER BY l_extendedprice DESC
        LIMIT 200
    """))

    queries.append(('Large Scan 2: Orders by Date', """
        SELECT
            o_orderkey,
            o_custkey,
            o_totalprice,
            o_orderdate,
            o_orderpriority
        FROM orders
        WHERE o_orderdate >= date '1996-01-01'
            AND o_orderdate < date '1997-01-01'
        ORDER BY o_orderdate DESC, o_totalprice DESC
        LIMIT 500
    """))

    queries.append(('Large Scan 3: Parts by Price', """
        SELECT
            p_partkey,
            p_name,
            p_brand,
            p_type,
            p_retailprice
        FROM part
        WHERE p_retailprice > 1000
        ORDER BY p_retailprice DESC, p_partkey
        LIMIT 300
    """))

    queries.append(('Large Scan 4: Customer Balance Ranking', """
        SELECT
            c.c_custkey,
            c.c_name,
            c.c_acctbal,
            c.c_mktsegment,
            n.n_name AS nation_name,
            r.r_name AS region_name
        FROM customer AS c
        JOIN nation AS n ON c.c_nationkey = n.n_nationkey
        JOIN region AS r ON n.n_regionkey = r.r_regionkey
        WHERE c.c_acctbal > 0
        ORDER BY c.c_acctbal DESC, c.c_name
        LIMIT 400
    """))

    queries.append(('Large Scan 5: Lineitem Quantity Sort', """
        SELECT
            l_orderkey,
            l_partkey,
            l_quantity,
            l_extendedprice,
            l_discount
        FROM lineitem
        WHERE l_shipdate >= date '1996-06-01'
            AND l_shipdate < date '1996-09-01'
            AND l_quantity > 40
        ORDER BY l_quantity DESC, l_extendedprice DESC
        LIMIT 250
    """))

    queries.append(('Large Scan 6: Recent Shipments', """
        SELECT
            l_orderkey,
            l_linenumber,
            l_shipdate,
            l_receiptdate,
            l_extendedprice * (1 - l_discount) as net_price
        FROM lineitem
        WHERE l_shipdate >= date '1998-06-01'
        ORDER BY l_shipdate DESC, net_price DESC
        LIMIT 300
    """))

    # ========================================================================
    # CATEGORY 7: Aggregation with HAVING
    # ========================================================================

    queries.append(('Having 1: Large Order Aggregates', """
        SELECT
            l_orderkey,
            SUM(l_quantity) as total_qty,
            SUM(l_extendedprice) as total_price,
            COUNT(*) as line_count
        FROM lineitem
        GROUP BY l_orderkey
        HAVING SUM(l_quantity) > 300
        ORDER BY total_price DESC
        LIMIT 100
    """))

    queries.append(('Having 2: High Volume Customers', """
        SELECT
            o_custkey,
            COUNT(*) as order_count,
            SUM(o_totalprice) as total_spent,
            AVG(o_totalprice) as avg_order_value
        FROM orders
        WHERE o_orderdate >= date '1996-01-01'
        GROUP BY o_custkey
        HAVING COUNT(*) > 15
        ORDER BY total_spent DESC
        LIMIT 50
    """))

    queries.append(('Having 3: Popular Parts by Brand', """
        SELECT
            p_brand,
            COUNT(*) as part_count,
            AVG(p_retailprice) as avg_price,
            MIN(p_retailprice) as min_price,
            MAX(p_retailprice) as max_price
        FROM part
        GROUP BY p_brand
        HAVING COUNT(*) > 50
        ORDER BY avg_price DESC
    """))

    queries.append(('Having 4: High Revenue Suppliers', """
        SELECT
            l_suppkey,
            COUNT(DISTINCT l_orderkey) as order_count,
            SUM(l_extendedprice * (1 - l_discount)) as total_revenue,
            AVG(l_quantity) as avg_quantity
        FROM lineitem
        WHERE l_shipdate >= date '1997-01-01'
        GROUP BY l_suppkey
        HAVING SUM(l_extendedprice * (1 - l_discount)) > 500000
        ORDER BY total_revenue DESC
        LIMIT 75
    """))

    queries.append(('Having 5: Part Categories with High Sales', """
        SELECT
            p_type,
            COUNT(DISTINCT l_orderkey) as order_count,
            SUM(l_quantity) as total_quantity,
            AVG(l_extendedprice) as avg_price
        FROM part
        JOIN lineitem ON p_partkey = l_partkey
        WHERE l_shipdate >= date '1996-01-01'
            AND l_shipdate < date '1997-01-01'
        GROUP BY p_type
        HAVING SUM(l_quantity) > 1000
        ORDER BY total_quantity DESC
        LIMIT 25
    """))

    queries.append(('Having 6: Customer Segments with Volume', """
        SELECT
            c_mktsegment,
            COUNT(DISTINCT c_custkey) as customer_count,
            COUNT(o_orderkey) as total_orders,
            AVG(o_totalprice) as avg_order_price
        FROM customer
        JOIN orders ON c_custkey = o_custkey
        WHERE o_orderdate >= date '1997-01-01'
        GROUP BY c_mktsegment
        HAVING COUNT(o_orderkey) > 1000
        ORDER BY total_orders DESC
    """))

    return queries

def main():
    """Main evaluation routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained QPP-Net model')
    parser.add_argument('--model', type=str, default='qpp_net_model.pt', help='Path to trained model')
    parser.add_argument('--no-actual', action='store_true', help='Skip actual execution time measurement')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for actual execution measurement')
    parser.add_argument('--save-results', type=str, default='evaluation_results.json', help='Path to save results JSON')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--query', '-q', type=str, action='append', dest='queries', help='Additional SQL query to evaluate (can be used multiple times)')
    parser.add_argument('--query-only', '-qo', action='store_true', help='Only evaluate the provided --query arguments, skip built-in test queries')

    args = parser.parse_args()

    print('=' * 80)
    print('QPP-NET MODEL EVALUATION')
    print('=' * 80)

    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)

    # Generate test queries
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

    # Run evaluation
    results = evaluator.evaluate_multiple_queries(
        test_queries,
        measure_actual=not args.no_actual,
        num_runs=args.runs
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    print(f'\nSaving results to {args.save_results}...')
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plots
    if not args.no_plots:
        try:
            evaluator.plot_results(results)
        except Exception as e:
            print(f'Warning: Could not generate plots: {e}')

    print('\n' + '=' * 80)
    print('EVALUATION COMPLETE!')
    print('=' * 80)

if __name__ == '__main__':
    main()
