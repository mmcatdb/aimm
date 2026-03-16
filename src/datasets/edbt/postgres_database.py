from typing_extensions import override
from common.config import DatasetName
from common.database import Database
from common.drivers import DriverType

class EdbtPostgresDatabase(Database[str]):
    def __init__(self):
        super().__init__(DatasetName.EDBT, DriverType.POSTGRES)

    @override
    def _generate_train_queries(self, num_queries: int):
        raise NotImplementedError(f'Training query generation not implemented for {self.id()} database.')

    @override
    def _generate_test_queries(self):
        # OLTP focused (mostly Postgres)

        person_id = 1
        self._test_query('Q3', 'Order history for a person (join via customer)', f'''
            SELECT o.order_id, o.ordered_at, o.status, o.total_cents, o.currency
            FROM "order" o
            JOIN customer c ON c.customer_id = o.customer_id
            WHERE c.person_id = '{person_id}'
            ORDER BY o.ordered_at DESC
            LIMIT 20
        ''')

        person_id = 1
        self._test_query('Q4', 'Order details view (now checks person via customer)', f'''
            SELECT
                o.order_id,
                o.ordered_at,
                o.status,
                oi.order_item_id,
                oi.product_id,
                COALESCE(oi.product_snapshot->>'title', p.title) AS product_title,
                oi.unit_price_cents,
                oi.quantity,
                oi.line_total_cents
            FROM "order" o
            JOIN customer c ON c.customer_id = o.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE c.person_id = '{person_id}'
            ORDER BY oi.order_item_id
        ''')

        person_id = 1
        product_id = 1
        self._test_query('Q6', 'How many times did this person buy this product? (via customer snapshots)', f'''
            SELECT COUNT(*)
            FROM customer c
            JOIN "order" o ON o.customer_id = c.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
                WHERE c.person_id = '{person_id}'
                  AND oi.product_id = '{product_id}'
                  AND o.status IN ('paid', 'shipped')
        ''')

        # OLAP focused (Postgres)

        seller_id = 1
        self._test_query('Q7', 'Seller daily revenue for last 30 days (Postgres, OLAP, medium weight)', f'''
            SELECT
                date_trunc('day', o.ordered_at) AS day,
                SUM(oi.line_total_cents) AS revenue_cents,
                COUNT(DISTINCT o.order_id) AS "order"
            FROM "order" o
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE p.seller_id = '{seller_id}'
              AND o.ordered_at >= now() - INTERVAL '30 days'
              AND o.status IN ('paid', 'shipped')
            GROUP BY 1
            ORDER BY 1
        ''')

        category_id = 1
        self._test_query('Q8', 'Top products by revenue inside one category, last 7 days (Postgres, OLAP, high weight in sale)', f'''
            SELECT
                oi.product_id,
                MAX(p.title) AS title,
                SUM(oi.line_total_cents) AS revenue_cents,
                SUM(oi.quantity) AS units
            FROM has_category hc
            JOIN product p ON p.product_id = hc.product_id
            JOIN order_item oi ON oi.product_id = hc.product_id
            JOIN "order" o ON o.order_id = oi.order_id
            WHERE hc.category_id = '{category_id}'
              AND o.ordered_at >= now() - INTERVAL '7 days'
              AND o.status IN ('paid', 'shipped')
            GROUP BY oi.product_id
            ORDER BY revenue_cents DESC
            LIMIT 50
        ''')

        self._test_query('Q9', 'Customer spend buckets (now per person)', f'''
            WITH spend AS (
                SELECT c.person_id, SUM(o.total_cents) AS spend_cents
                FROM customer c
                JOIN "order" o ON o.customer_id = c.customer_id
                WHERE o.ordered_at >= now() - INTERVAL '90 days'
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
        ''')

        self._test_query('Q10', 'Fraud-ish pattern (now per person)', f'''
            SELECT
                c.person_id,
                COUNT(DISTINCT p.seller_id) AS distinct_sellers,
                COUNT(DISTINCT o.order_id) AS "order"
            FROM customer c
            JOIN "order" o ON o.customer_id = c.customer_id
            JOIN order_item oi ON oi.order_id = o.order_id
            JOIN product p ON p.product_id = oi.product_id
            WHERE o.ordered_at >= now() - INTERVAL '24 hours'
              AND o.status IN ('paid', 'shipped')
            GROUP BY c.person_id
            HAVING COUNT(DISTINCT p.seller_id) >= 10
            ORDER BY distinct_sellers DESC
            LIMIT 200
        ''')

        # This is designed to be heavy if you do it wrong, but fast if you limit scope. Good for optimizer tests.
        # Input: a temp table hot_products(product_id) with maybe top 1% products.
        # Output: co-buy counts for pairs, last 7 days, only when at least one side is hot.
        # self._test_query('Q11', 'Rebuild "also bought" pairs for hot products only (Postgres, OLAP, sale-specific)', '''
        #     WITH recent AS (
        #         SELECT oi.order_id, oi.product_id
        #         FROM order_item oi
        #         JOIN "order" o ON o.order_id = oi.order_id
        #         WHERE o.ordered_at >= now() - INTERVAL '7 days'
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
        # ''')

        # Document focused (MongoDB)
        # These are built to avoid joins at read time. Put "product page bundle" in one document.

        product_id = 1
        self._test_query('Q12', 'Product page read (Mongo, OLTP read-heavy, very high weight in sale)', f'''
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
                    LIMIT 5
                ) r
                GROUP BY r.product_id
            ) tr ON tr.product_id = p.product_id
            WHERE p.product_id = '{product_id}'
              AND p.is_active = TRUE
        ''')

        product_ids = [ 1, 2, 3 ]
        product_ids_string = ', '.join([f"'{id}'" for id in product_ids])
        self._test_query('Q13', 'Bulk fetch product pages for a feed (Mongo, OLTP read-heavy, high weight)', f'''
            SELECT
                p.product_id,
                p.title,
                p.price_cents,
                p.currency,
                jsonb_build_object(
                    'seller_id', s.seller_id,
                    'display_name', s.display_name
                ) AS seller,
                jsonb_build_object(
                    'avg', COALESCE(rs.avg_rating, 0),
                    'count', COALESCE(rs.review_count, 0)
                ) AS rating_summary
            FROM product p
            JOIN seller s ON s.seller_id = p.seller_id
            LEFT JOIN (
                SELECT
                    r.product_id,
                    AVG(r.rating)::float8 AS avg_rating,
                    COUNT(*)::int AS review_count
                FROM review r
                WHERE r.product_id IN ({product_ids_string})
                GROUP BY r.product_id
            ) rs ON rs.product_id = p.product_id
            WHERE p.product_id IN ({product_ids_string})
              AND p.is_active = TRUE
            LIMIT 200
        ''')

        # Graph focused (Neo4j)

        person_id = 1
        self._test_query('Q15', 'Who should I follow? (User -> Person)', f'''
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
            LIMIT 50
        ''')

        product_id = 1
        self._test_query('Q16', 'Neo4j replacement for old "SIMILAR" query (since similar is removed) "People also bought" using shared orders:', f'''
            SELECT
                oi2.product_id,
                COUNT(*) AS co_buy
            FROM order_item oi1
            JOIN order_item oi2 ON oi1.order_id = oi2.order_id
            JOIN "order" o ON o.order_id = oi1.order_id
            WHERE oi1.product_id = '{product_id}'
              AND oi2.product_id <> '{product_id}'
              AND o.status IN ('paid', 'shipped')
            GROUP BY oi2.product_id
            ORDER BY co_buy DESC
            LIMIT 20
        ''')

        person_id = 1
        self._test_query('Q17', 'Personalized feed candidates (User -> Person)', f'''
            SELECT
                hc.product_id,
                SUM(hi.strength) AS interest_score
            FROM has_interest hi
            JOIN has_category hc ON hc.category_id = hi.category_id
            JOIN product p ON p.product_id = hc.product_id
            WHERE hi.person_id = '{person_id}'
              AND p.is_active = TRUE
            GROUP BY hc.product_id
            ORDER BY interest_score DESC
            LIMIT 200
        ''')

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
