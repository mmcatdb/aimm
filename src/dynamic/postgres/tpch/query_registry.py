from core.drivers import DriverType
from core.query import QueryRegistry, query


class TpchPostgresQueryRegistry(QueryRegistry[str]):
    def __init__(self):
        super().__init__(DriverType.POSTGRES, 'tpch')

    def _customer_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(30_000 * scale))

    def _order_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(1_200_000 * scale))

    def _part_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(40_000 * scale))

    def _supplier_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(2_000 * scale))

    def _quoted_choice(self, name: str, values: list[str]) -> str:
        value = self._param_choice(name, values)
        return f"'{value}'"

    @query('customer-orders-by-segment', 'Customer order totals by market segment')
    def customer_orders_by_segment(self) -> str:
        segment = self._quoted_choice('segment', ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'HOUSEHOLD', 'MACHINERY'])
        return f"""
            SELECT c.c_mktsegment, COUNT(*) AS order_count, SUM(o.o_totalprice) AS total_price
            FROM customer c
            JOIN orders o ON o.o_custkey = c.c_custkey
            WHERE c.c_mktsegment = {segment}
            GROUP BY c.c_mktsegment
        """

    @query('lineitem-shipdate', 'Lineitem revenue for a ship-date range')
    def lineitem_shipdate(self) -> str:
        year = self._param_int('year', 1993, 1997)
        return f"""
            SELECT l_returnflag, l_linestatus,
                   SUM(l_extendedprice * (1 - l_discount)) AS revenue,
                   COUNT(*) AS line_count
            FROM lineitem
            WHERE l_shipdate >= DATE '{year}-01-01'
              AND l_shipdate < DATE '{year + 1}-01-01'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        """

    @query('supplier-parts', 'Parts supplied by a supplier')
    def supplier_parts(self) -> str:
        supplier_id = self._param_int('supplier_id', 1, self._supplier_max())
        return f"""
            SELECT s.s_name, p.p_name, ps.ps_availqty, ps.ps_supplycost
            FROM supplier s
            JOIN partsupp ps ON ps.ps_suppkey = s.s_suppkey
            JOIN part p ON p.p_partkey = ps.ps_partkey
            WHERE s.s_suppkey = {supplier_id}
            ORDER BY ps.ps_supplycost DESC
            LIMIT 50
        """

    @query('customer-network-orders', 'Known customers and their recent orders')
    def customer_network_orders(self) -> str:
        customer_id = self._param_int('customer_id', 1, self._customer_max())
        return f"""
            SELECT k.k_custkey2, COUNT(o.o_orderkey) AS order_count, SUM(o.o_totalprice) AS total_price
            FROM knows k
            JOIN orders o ON o.o_custkey = k.k_custkey2
            WHERE k.k_custkey1 = {customer_id}
            GROUP BY k.k_custkey2
            ORDER BY total_price DESC NULLS LAST
            LIMIT 25
        """

    @query('part-order-volume', 'Order volume for a part')
    def part_order_volume(self) -> str:
        part_id = self._param_int('part_id', 1, self._part_max())
        return f"""
            SELECT l.l_partkey, COUNT(*) AS lines, SUM(l.l_quantity) AS quantity
            FROM lineitem l
            WHERE l.l_partkey = {part_id}
            GROUP BY l.l_partkey
        """

    @query('order-detail', 'Single order with line items')
    def order_detail(self) -> str:
        order_id = self._param_int('order_id', 1, self._order_max())
        return f"""
            WITH selected_order AS (
                SELECT o_orderkey
                FROM orders
                WHERE o_orderkey >= {order_id}
                ORDER BY o_orderkey
                LIMIT 1
            )
            SELECT o.o_orderkey, o.o_orderdate, o.o_totalprice, l.l_linenumber,
                   p.p_name, l.l_quantity, l.l_extendedprice
            FROM selected_order so
            JOIN orders o ON o.o_orderkey = so.o_orderkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            JOIN part p ON p.p_partkey = l.l_partkey
            ORDER BY l.l_linenumber
        """


def export():
    return TpchPostgresQueryRegistry()
