from abc import ABC, abstractmethod
from typing import Any

class TpchDAO(ABC):
    #region Selection, Projection, Source (of data)

    @abstractmethod
    def get_all_lineitems(self) -> Any:
        """
        A1) Non-Indexed Columns

        This query selects all records from the lineitem table
        ```sql
        SELECT * FROM lineitem
        ```
        """
        pass

    @abstractmethod
    def get_orders_by_daterange(self, start_date, end_date) -> Any:
        """
        A2) Non-Indexed Columns — Range Query

        This query selects all records from the orders table where the order date is between '1996-01-01' and '1996-12-31'
        ```sql
        SELECT * FROM orders
        WHERE o_orderdate BETWEEN '1996-01-01' AND '1996-12-31'
        ```
        """
        pass

    @abstractmethod
    def get_all_customers(self) -> Any:
        """
        A3) Indexed Columns

        This query selects all records from the customer table
        ```sql
        SELECT * FROM customer
        ```
        """
        pass

    @abstractmethod
    def get_orders_by_keyrange(self, start_key, end_key) -> Any:
        """
        A4) Indexed Columns — Range Query

        This query selects all records from the orders table where the order key is between 1000 and 50000
        ```sql
        SELECT * FROM orders
        WHERE o_orderkey BETWEEN 1000 AND 50000
        ```
        """
        pass

    #endregion
    #region Aggregation

    @abstractmethod
    def count_orders_by_month(self) -> Any:
        """
        B1) COUNT

        This query counts the number of orders grouped by order month
        ```sql
        SELECT COUNT(o.o_orderkey) AS order_count, DATE_FORMAT(o.o_orderdate, '%Y-%m') AS order_month
        FROM orders o
        GROUP BY order_month
        ```
        """
        pass

    @abstractmethod
    def get_max_price_by_ship_month(self) -> Any:
        """
        B2) MAX

        This query finds the maximum extended price from the lineitem table grouped by ship month
        ```sql
        SELECT DATE_FORMAT(l.l_shipdate, '%Y-%m') AS ship_month, MAX(l.l_extendedprice) AS max_price
        FROM lineitem l
        GROUP BY ship_month
        ```
        """
        pass

    @abstractmethod
    def get_all_parts(self) -> Any:
        """P1) List all parts."""
        pass

    @abstractmethod
    def get_parts_by_size_range(self, min_size, max_size) -> Any:
        """P2) Range filter on p_size."""
        pass

    @abstractmethod
    def get_all_suppliers(self) -> Any:
        """S1) List all suppliers."""
        pass

    @abstractmethod
    def get_suppliers_by_nation(self, nation_key) -> Any:
        """S2) Filter suppliers by nation key."""
        pass

    @abstractmethod
    def get_partsupp_for_part(self, partkey) -> Any:
        """PS1) All supplier rows for a given part."""
        pass

    @abstractmethod
    def get_lowest_cost_supplier_for_part(self, partkey) -> Any:
        """PS2) Cheapest supplier for a part."""
        pass

    @abstractmethod
    def count_suppliers_per_part(self) -> Any:
        """AGG1) Number of suppliers per part (cardinality of part->supplier)."""
        pass

    @abstractmethod
    def avg_supplycost_by_part_size(self) -> Any:
        """AGG2) Average supply cost grouped by part size (join part + partsupp)."""
        pass

    #endregion
