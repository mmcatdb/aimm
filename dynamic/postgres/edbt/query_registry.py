from typing_extensions import override
from core.drivers import DriverType
from ...common.edbt.query_registry import EdbtQueryRegistry

def export():
    return PostgresEdbtQueryRegistry()

class PostgresEdbtQueryRegistry(EdbtQueryRegistry[str]):

    def __init__(self):
        super().__init__(DriverType.POSTGRES)

    # OLTP focused (mostly Postgres)

    @override
    def _order_history_for_person(self, person_id):
        return f'''
            SELECT
                o.order_id,
                o.ordered_at,
                o.status,
                o.total_cents,
                o.currency
            FROM "order" o
            JOIN customer c ON c.customer_id = o.customer_id
            WHERE c.person_id = '{person_id}'
            ORDER BY o.ordered_at DESC
        '''

    @override
    def _order_details(self, person_id):
        return f'''
            SELECT
                o.order_id,
                o.ordered_at,
                o.status,
                oi.order_item_id,
                oi.product_id,
                p.title,
                oi.unit_price_cents,
                oi.quantity,
                oi.line_total_cents
            FROM "order" o
            JOIN customer c ON c.customer_id = o.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE c.person_id = '{person_id}'
            ORDER BY oi.order_item_id
        '''

    @override
    def _product_purchases_for_person(self, person_ids, product_ids):
        return f'''
            SELECT COUNT(*)
            FROM customer c
            JOIN "order" o ON o.customer_id = c.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
            WHERE c.person_id IN ({person_ids})
                AND oi.product_id IN ({product_ids})
                AND o.status IN ('paid', 'shipped')
        '''

    # OLAP focused (Postgres)

    @override
    def _seller_daily_revenue(self, date, seller_ids):
        return f'''
            SELECT
                date_trunc('day', o.ordered_at) AS day,
                SUM(oi.line_total_cents) AS revenue_cents,
                COUNT(DISTINCT o.order_id) AS "order"
            FROM "order" o
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE p.seller_id IN ({seller_ids})
                AND o.ordered_at >= '{date}'
                AND o.status IN ('paid', 'shipped')
            GROUP BY 1
            ORDER BY 1
        '''

    @override
    def _top_products_by_revenue(self, date, category_ids):
        return f'''
            SELECT
                oi.product_id,
                p.title,
                SUM(oi.line_total_cents) AS revenue_cents,
                SUM(oi.quantity) AS units
            FROM has_category hc
            JOIN product p ON p.product_id = hc.product_id
            JOIN order_item oi ON oi.product_id = hc.product_id
            JOIN "order" o ON o.order_id = oi.order_id
            WHERE hc.category_id IN ({category_ids})
                AND o.ordered_at >= '{date}'
                AND o.status IN ('paid', 'shipped')
            GROUP BY oi.product_id, p.title
            ORDER BY revenue_cents DESC
            LIMIT 200
        '''

    @override
    def _customer_spend_buckets(self, date):
        return f'''
            WITH spend AS (
                SELECT c.person_id, SUM(o.total_cents) AS spend_cents
                FROM customer c
                JOIN "order" o ON o.customer_id = c.customer_id
                WHERE o.ordered_at >= '{date}'
                    AND o.status IN ('paid', 'shipped')
                GROUP BY c.person_id
            )
            SELECT
                CASE
                    WHEN spend_cents < 5000 THEN 'low'
                    WHEN spend_cents < 20000 THEN 'mid'
                    ELSE 'high'
                END AS bucket,
                COUNT(*) AS persons
            FROM spend
            GROUP BY 1
            ORDER BY 1
        '''

    @override
    def _fraud_pattern(self, date, distinct_sellers_threshold):
        return f'''
            SELECT
                c.person_id,
                COUNT(DISTINCT p.seller_id) AS distinct_sellers,
                COUNT(DISTINCT o.order_id) AS "order"
            FROM customer c
            JOIN "order" o ON o.customer_id = c.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE o.ordered_at >= '{date}'
                AND o.status IN ('paid', 'shipped')
            GROUP BY c.person_id
            HAVING COUNT(DISTINCT p.seller_id) >= {distinct_sellers_threshold}
            ORDER BY distinct_sellers DESC
            LIMIT 200
        '''

    @override
    def _who_to_follow(self, person_id):
        return f'''
            SELECT
                f2.from_id AS person_id,
                COUNT(*) AS paths
            FROM follows f1
            JOIN follows f2 ON f1.from_id = f2.to_id
            WHERE f1.to_id = '{person_id}'
                AND f2.from_id <> '{person_id}'
                AND NOT EXISTS (
                    SELECT 1
                    FROM follows direct
                    WHERE direct.to_id = '{person_id}'
                        AND direct.from_id = f2.from_id
                )
            GROUP BY f2.from_id
            ORDER BY paths DESC
            LIMIT 200
        '''


    @override
    def _personalized_feed_candidates(self, person_ids):
        return f'''
            SELECT
                hc.product_id,
                SUM(hi.strength) AS interest_score
            FROM has_interest hi
            JOIN has_category hc ON hc.category_id = hi.category_id
            JOIN product p ON p.product_id = hc.product_id
            WHERE hi.person_id IN ({person_ids})
                AND p.is_active = TRUE
            GROUP BY hc.product_id
            ORDER BY interest_score DESC
            LIMIT 200
        '''

    # This is designed to be heavy if we do it wrong, but fast if we limit scope. Good for optimizer tests.
    # Input: a temp table hot_products(product_id) with maybe top 1% products.
    # Output: co-buy counts for pairs, last 7 days, only when at least one side is hot.
    # 'Q11',
    # 'Rebuild "also bought" pairs for hot products only (Postgres, OLAP, sale-specific)', '''
    #     WITH recent AS (
    #         SELECT oi.order_id, oi.product_id
    #         FROM order_item oi
    #         JOIN "order" o ON o.order_id = oi.order_id
    #         WHERE o.ordered_at >= '{self._param_now()}'::TIMESTAMPTZ - INTERVAL '7 days'
    #           AND o.status IN ('paid', 'shipped')
    #     ),
    #     pairs AS (
    #         SELECT
    #             LEAST(a.product_id, b.product_id) AS product_id_a,
    #             GREATEST(a.product_id, b.product_id) AS product_id_b,
    #             COUNT(*) AS co_buy_count
    #         FROM recent a
    #         JOIN recent b
    #             ON a.order_id = b.order_id
    #            AND a.product_id < b.product_id
    #         LEFT JOIN hot_products hp_a ON hp_a.product_id = a.product_id
    #         LEFT JOIN hot_products hp_b ON hp_b.product_id = b.product_id
    #         WHERE hp_a.product_id IS NOT NULL
    #         OR hp_b.product_id IS NOT NULL
    #         GROUP BY 1, 2
    #     )
    #     SELECT product_id_a, product_id_b, co_buy_count
    #     FROM pairs
    #     WHERE co_buy_count >= 5
    #     ORDER BY co_buy_count DESC
    #     LIMIT 50000
    # '''

    # Document focused (MongoDB)
    # These are built to avoid joins at read time. Put "product page bundle" in one document.

    @override
    def _product_page_read(self, product_id):
        return f'''
            SELECT
                p.product_id,
                p.title,
                p.price_cents,
                p.currency,
                p.stock_qty,
                jsonb_build_object(
                    'seller_id', s.seller_id,
                    'display_name', s.display_name
                ) AS seller,
                jsonb_build_object(
                    'avg', COALESCE(rs.avg_rating, 0),
                    'count', COALESCE(rs.review_count, 0)
                ) AS rating_summary,
                COALESCE(tr.top_reviews, '[]'::jsonb) AS top_reviews
            FROM product p
            JOIN seller s ON s.seller_id = p.seller_id
            LEFT JOIN (
                SELECT
                    r.product_id,
                    AVG(r.rating)::float8 AS avg_rating,
                    COUNT(*)::int AS review_count
                FROM review r
                WHERE r.product_id = '{product_id}'
                GROUP BY r.product_id
            ) rs ON rs.product_id = p.product_id
            LEFT JOIN (
                SELECT
                    r.product_id,
                    jsonb_agg(
                    jsonb_build_object(
                        'customer_id', r.customer_id,
                        'rating', r.rating,
                        'title', r.title,
                        'body', r.body,
                        'created_at', r.created_at,
                        'helpful_votes', r.helpful_votes
                    )
                    ORDER BY r.helpful_votes DESC, r.created_at DESC
                    ) AS top_reviews
                FROM (
                    SELECT *
                    FROM review
                    WHERE product_id = '{product_id}'
                    ORDER BY helpful_votes DESC, created_at DESC
                    LIMIT 50
                ) r
                GROUP BY r.product_id
            ) tr ON tr.product_id = p.product_id
            WHERE p.product_id = '{product_id}'
                AND p.is_active = TRUE
        '''

    # Bulk fetch product pages for a feed (Mongo, OLTP read-heavy, high weight)
    # def _bulk_fetch_product_pages(self):
    #     product_ids = self._param_product_ids(5, 20)

    #     return f'''
    #         SELECT
    #             p.product_id,
    #             p.title,
    #             p.price_cents,
    #             p.currency,
    #             jsonb_build_object(
    #                 'seller_id', s.seller_id,
    #                 'display_name', s.display_name
    #             ) AS seller,
    #             jsonb_build_object(
    #                 'avg', COALESCE(rs.avg_rating, 0),
    #                 'count', COALESCE(rs.review_count, 0)
    #             ) AS rating_summary
    #         FROM product p
    #         JOIN seller s ON s.seller_id = p.seller_id
    #         LEFT JOIN (
    #             SELECT
    #                 r.product_id,
    #                 AVG(r.rating)::float8 AS avg_rating,
    #                 COUNT(*)::int AS review_count
    #             FROM review r
    #             WHERE r.product_id IN ({product_ids})
    #             GROUP BY r.product_id
    #         ) rs ON rs.product_id = p.product_id
    #         WHERE p.product_id IN ({product_ids})
    #             AND p.is_active = TRUE
    #         LIMIT 200
    #     '''

    @override
    def _people_also_bought(self, product_ids):
        return f'''
            SELECT
                oi2.product_id,
                COUNT(*) AS co_buy
            FROM order_item oi1
            JOIN order_item oi2 ON oi1.order_id = oi2.order_id
            JOIN "order" o ON o.order_id = oi1.order_id
            WHERE oi1.product_id IN ({product_ids})
                AND oi2.product_id NOT IN ({product_ids})
                AND o.status IN ('paid', 'shipped')
            GROUP BY oi2.product_id
            ORDER BY co_buy DESC
            LIMIT 20
        '''

    @override
    def _order_revenue_by_status_currency(self, statuses):
        return f'''
            SELECT
                o.status,
                o.currency,
                COUNT(*)::int AS orders,
                SUM(o.total_cents)::bigint AS revenue_cents,
                AVG(o.total_cents)::float8 AS avg_order_cents
            FROM "order" o
            WHERE o.status IN ({statuses})
            GROUP BY o.status, o.currency
            ORDER BY o.status, o.currency
        '''

    @override
    def _product_inventory_by_price_band(self, max_price_cents, min_stock_qty):
        return f'''
            WITH filtered AS (
                SELECT
                    CASE
                        WHEN p.price_cents < 1000 THEN 0
                        WHEN p.price_cents < 5000 THEN 1
                        WHEN p.price_cents < 20000 THEN 2
                        ELSE 3
                    END AS price_band,
                    p.currency,
                    p.stock_qty,
                    p.price_cents
                FROM product p
                WHERE p.is_active = TRUE
                    AND p.price_cents <= {max_price_cents}
                    AND p.stock_qty >= {min_stock_qty}
            )
            SELECT
                price_band,
                currency,
                COUNT(*)::int AS products,
                SUM(stock_qty)::bigint AS stock_qty,
                AVG(price_cents)::float8 AS avg_price_cents
            FROM filtered
            GROUP BY price_band, currency
            ORDER BY currency, price_band
        '''

    @override
    def _review_rating_distribution(self, min_helpful_votes):
        return f'''
            SELECT
                r.rating,
                COUNT(*)::int AS reviews,
                AVG(r.helpful_votes)::float8 AS avg_helpful_votes
            FROM review r
            WHERE r.helpful_votes >= {min_helpful_votes}
            GROUP BY r.rating
            ORDER BY r.rating
        '''

    @override
    def _customer_snapshot_activity(self, country_codes):
        return f'''
            SELECT
                c.country_code,
                c.is_active,
                COUNT(*)::int AS customers,
                COUNT(DISTINCT c.person_id)::int AS persons
            FROM customer c
            WHERE c.country_code IN ({country_codes})
            GROUP BY c.country_code, c.is_active
            ORDER BY c.country_code, c.is_active
        '''

    @override
    def _seller_activity_rollup_by_country(self, country_codes):
        return f'''
            SELECT
                s.country_code,
                s.is_active,
                COUNT(*)::int AS sellers
            FROM seller s
            WHERE s.country_code IN ({country_codes})
            GROUP BY s.country_code, s.is_active
            ORDER BY s.country_code, s.is_active
        '''

    @override
    def _line_item_quantity_distribution(self, min_unit_price_cents):
        return f'''
            SELECT
                oi.quantity,
                COUNT(*)::int AS items,
                SUM(oi.quantity)::bigint AS units,
                SUM(oi.line_total_cents)::bigint AS revenue_cents,
                AVG(oi.unit_price_cents)::float8 AS avg_unit_price_cents
            FROM order_item oi
            WHERE oi.unit_price_cents >= {min_unit_price_cents}
            GROUP BY oi.quantity
            ORDER BY oi.quantity
        '''

    @override
    def _seller_sales_summary(self, date, seller_ids):
        return f'''
            SELECT
                p.seller_id,
                COUNT(DISTINCT o.order_id)::int AS orders,
                COUNT(*)::int AS items,
                SUM(oi.quantity)::bigint AS units,
                SUM(oi.line_total_cents)::bigint AS revenue_cents
            FROM "order" o
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE p.seller_id IN ({seller_ids})
                AND o.ordered_at >= '{date}'
                AND o.status IN ('paid', 'shipped')
            GROUP BY p.seller_id
            ORDER BY revenue_cents DESC, p.seller_id
            LIMIT 200
        '''

    @override
    def _customer_country_order_status(self, date, country_codes):
        return f'''
            SELECT
                c.country_code,
                o.status,
                COUNT(*)::int AS orders,
                SUM(o.total_cents)::bigint AS revenue_cents,
                AVG(o.total_cents)::float8 AS avg_order_cents
            FROM customer c
            JOIN "order" o ON o.customer_id = c.customer_id
            WHERE c.country_code IN ({country_codes})
                AND o.ordered_at >= '{date}'
            GROUP BY c.country_code, o.status
            ORDER BY c.country_code, o.status
        '''

    @override
    def _product_review_summary(self, product_ids, min_helpful_votes):
        return f'''
            SELECT
                r.product_id,
                COUNT(*)::int AS reviews,
                AVG(r.rating)::float8 AS avg_rating,
                SUM(r.helpful_votes)::bigint AS helpful_votes,
                MAX(r.helpful_votes)::int AS max_helpful_votes
            FROM review r
            WHERE r.product_id IN ({product_ids})
                AND r.helpful_votes >= {min_helpful_votes}
            GROUP BY r.product_id
            ORDER BY reviews DESC, r.product_id
            LIMIT 200
        '''

    @override
    def _category_interest_summary(self, category_ids, min_strength):
        return f'''
            SELECT
                hi.category_id,
                COUNT(DISTINCT hi.person_id)::int AS interested_persons,
                AVG(hi.strength)::float8 AS avg_strength,
                MAX(hi.strength)::int AS max_strength
            FROM has_interest hi
            JOIN person p ON p.person_id = hi.person_id
            WHERE hi.category_id IN ({category_ids})
                AND hi.strength >= {min_strength}
                AND p.is_active = TRUE
            GROUP BY hi.category_id
            ORDER BY interested_persons DESC, hi.category_id
            LIMIT 200
        '''

    @override
    def _category_catalog_summary(self, category_ids, max_price_cents, min_stock_qty):
        return f'''
            SELECT
                hc.category_id,
                COUNT(DISTINCT p.product_id)::int AS products,
                SUM(p.stock_qty)::bigint AS stock_qty,
                AVG(p.price_cents)::float8 AS avg_price_cents
            FROM has_category hc
            JOIN product p ON p.product_id = hc.product_id
            WHERE hc.category_id IN ({category_ids})
                AND p.is_active = TRUE
                AND p.price_cents <= {max_price_cents}
                AND p.stock_qty >= {min_stock_qty}
            GROUP BY hc.category_id
            ORDER BY products DESC, hc.category_id
            LIMIT 200
        '''

    @override
    def _seller_catalog_health(self, seller_ids):
        return f'''
            SELECT
                p.seller_id,
                COUNT(*)::int AS products,
                SUM(CASE WHEN p.is_active THEN 1 ELSE 0 END)::int AS active_products,
                SUM(CASE WHEN p.is_active THEN 0 ELSE 1 END)::int AS inactive_products,
                SUM(p.stock_qty)::bigint AS stock_qty,
                AVG(p.price_cents)::float8 AS avg_price_cents
            FROM product p
            WHERE p.seller_id IN ({seller_ids})
            GROUP BY p.seller_id
            ORDER BY products DESC, p.seller_id
            LIMIT 200
        '''

    @override
    def _follow_country_rollup(self, country_codes):
        return f'''
            SELECT
                p_from.country_code AS from_country,
                p_to.country_code AS to_country,
                COUNT(*)::int AS edges,
                SUM(CASE WHEN p_to.is_active THEN 1 ELSE 0 END)::int AS active_targets
            FROM follows f
            JOIN person p_from ON p_from.person_id = f.from_id
            JOIN person p_to ON p_to.person_id = f.to_id
            WHERE p_from.country_code IN ({country_codes})
            GROUP BY p_from.country_code, p_to.country_code
            ORDER BY edges DESC, from_country, to_country
            LIMIT 200
        '''

    # One "multi-db" query (on purpose)

    # Q18) Feed ranking in Neo4j, then page fetch in Mongo (Cross DB, high weight in sale)

    # Part A (Neo4j): get 200 product ids

    # MATCH (p:Person {person_id: $personId})-[hi:HAS_INTEREST]->(c:Category)<-[:HAS_CATEGORY]-(pr:Product)
    # WHERE pr.is_active = true
    # RETURN pr.product_id AS product_id
    # ORDER BY SUM(hi.strength) DESC
    # LIMIT 200

    # Part B (Mongo): fetch those 200 docs

    # db.product_page.find(
    # { _id: { $in: productIds.map(x => NumberLong(x)) }, is_active: true },
    # { title: 1, price_cents: 1, currency: 1, seller: 1, rating_summary: 1, sale_score_1h: 1 }
    # )
