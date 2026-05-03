from core.drivers import DriverType
from core.query import QueryRegistry, query


class EdbtPostgresQueryRegistry(QueryRegistry[str]):
    def __init__(self):
        super().__init__(DriverType.POSTGRES, 'edbt')

    def _person_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(50_000 * scale))

    def _customer_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(200_000 * scale))

    def _product_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(20_000 * scale))

    def _category_max(self) -> int:
        scale = self._scale or 1.0
        return max(1, round(2_000 * (scale ** 0.7)))

    def _quoted_choice(self, name: str, values: list[str]) -> str:
        value = self._param_choice(name, values)
        return f"'{value}'"

    @query('customer-orders', 'Orders for one customer')
    def customer_orders(self) -> str:
        customer_id = self._param_int('customer_id', 1, self._customer_max())
        return f"""
            SELECT o.order_id, o.ordered_at, o.status, o.total_cents,
                   COUNT(oi.order_item_id) AS item_count
            FROM "order" o
            LEFT JOIN order_item oi ON oi.order_id = o.order_id
            WHERE o.customer_id = {customer_id}
            GROUP BY o.order_id, o.ordered_at, o.status, o.total_cents
            ORDER BY o.ordered_at DESC
            LIMIT 50
        """

    @query('active-products-by-category', 'Active products in a category')
    def active_products_by_category(self) -> str:
        category_id = self._param_int('category_id', 1, self._category_max())
        return f"""
            SELECT p.product_id, p.title, p.price_cents, s.display_name
            FROM has_category hc
            JOIN product p ON p.product_id = hc.product_id
            JOIN seller s ON s.seller_id = p.seller_id
            WHERE hc.category_id = {category_id}
              AND p.is_active = TRUE
            ORDER BY p.price_cents DESC
            LIMIT 100
        """

    @query('product-reviews', 'Reviews for one product')
    def product_reviews(self) -> str:
        product_id = self._param_int('product_id', 1, self._product_max())
        return f"""
            SELECT r.review_id, r.rating, r.helpful_votes, c.customer_id, c.country_code
            FROM review r
            JOIN customer c ON c.customer_id = r.customer_id
            WHERE r.product_id = {product_id}
            ORDER BY r.helpful_votes DESC, r.created_at DESC
            LIMIT 100
        """

    @query('person-feed', 'Products from followed people interests')
    def person_feed(self) -> str:
        person_id = self._param_int('person_id', 1, self._person_max())
        return f"""
            SELECT p.product_id, p.title, p.price_cents, COUNT(*) AS signal_count
            FROM follows f
            JOIN has_interest hi ON hi.person_id = f.to_id
            JOIN has_category hc ON hc.category_id = hi.category_id
            JOIN product p ON p.product_id = hc.product_id
            WHERE f.from_id = {person_id}
              AND p.is_active = TRUE
            GROUP BY p.product_id, p.title, p.price_cents
            ORDER BY signal_count DESC, p.price_cents DESC
            LIMIT 50
        """

    @query('seller-performance', 'Seller revenue and volume')
    def seller_performance(self) -> str:
        status = self._quoted_choice('status', ['paid', 'shipped'])
        return f"""
            SELECT s.seller_id, s.display_name,
                   COUNT(DISTINCT o.order_id) AS orders,
                   SUM(oi.line_total_cents) AS revenue_cents
            FROM seller s
            JOIN product p ON p.seller_id = s.seller_id
            JOIN order_item oi ON oi.product_id = p.product_id
            JOIN "order" o ON o.order_id = oi.order_id
            WHERE o.status = {status}
            GROUP BY s.seller_id, s.display_name
            ORDER BY revenue_cents DESC NULLS LAST
            LIMIT 50
        """

    @query('category-interest', 'People interested in a category')
    def category_interest(self) -> str:
        category_id = self._param_int('category_id', 1, self._category_max())
        return f"""
            SELECT p.country_code, COUNT(*) AS people, AVG(hi.strength) AS avg_strength
            FROM has_interest hi
            JOIN person p ON p.person_id = hi.person_id
            WHERE hi.category_id = {category_id}
              AND p.is_active = TRUE
            GROUP BY p.country_code
            ORDER BY people DESC
        """


def export():
    return EdbtPostgresQueryRegistry()
