from typing_extensions import override
from core.drivers import DriverType
from ...common.edbt.query_registry import EdbtQueryRegistry

def export():
    return Neo4jEdbtQueryRegistry()

class Neo4jEdbtQueryRegistry(EdbtQueryRegistry[str]):

    def __init__(self):
        super().__init__(DriverType.NEO4J)

    # OLTP focused (mostly Postgres)

    @override
    def _order_history_for_person(self, person_id):
        return f'''
            MATCH (p:Person {{person_id: {person_id}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            RETURN
                o.order_id AS order_id,
                o.ordered_at AS ordered_at,
                o.status AS status,
                o.total_cents AS total_cents,
                o.currency AS currency
            ORDER BY o.ordered_at DESC
        '''

    @override
    def _order_details(self, person_id):
        return f'''
            MATCH (p:Person {{person_id: {person_id}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            MATCH (o)-[it:HAS_ITEM]->(pr:Product)
            RETURN
                o.order_id AS order_id,
                o.ordered_at AS ordered_at,
                o.status AS status,
                it.order_item_id AS order_item_id,
                pr.product_id AS product_id,
                pr.title AS product_title,
                it.unit_price_cents AS unit_price_cents,
                it.quantity AS quantity,
                (it.unit_price_cents * it.quantity) AS line_total_cents
            ORDER BY order_item_id
        '''

    @override
    def _product_purchases_for_person(self, person_ids, product_ids):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
            MATCH (o)-[:HAS_ITEM]->(pr:Product)
            WHERE p.person_id IN [{person_ids}]
                AND pr.product_id IN [{product_ids}]
            RETURN COUNT(*) AS purchase_count
        '''

    # OLAP focused (Postgres)

    @override
    def _seller_daily_revenue(self, date, seller_ids):
        return f'''
            MATCH (s:Seller)-[:OFFERS]->(pr:Product)
            MATCH (o:Order)-[it:HAS_ITEM]->(pr)
            WHERE s.seller_id IN [{seller_ids}]
                AND o.ordered_at >= datetime('{date}')
                AND o.status IN ['paid', 'shipped']
            WITH
                date(o.ordered_at) AS day,
                (it.unit_price_cents * it.quantity) AS line_total,
                o.order_id AS order_id
            RETURN
                day AS day,
                sum(line_total) AS revenue_cents,
                count(DISTINCT order_id) AS orders
            ORDER BY day
        '''

    @override
    def _top_products_by_revenue(self, date, category_ids):
        return f'''
            MATCH (c:Category)<-[:HAS_CATEGORY]-(pr:Product)
            MATCH (o:Order)-[it:HAS_ITEM]->(pr)
            WHERE c.category_id IN [{category_ids}]
                AND o.ordered_at >= datetime('{date}')
                AND o.status IN ['paid', 'shipped']
            WITH
                pr,
                sum(it.unit_price_cents * it.quantity) AS revenue_cents,
                sum(it.quantity) AS units
            RETURN
                pr.product_id AS product_id,
                pr.title AS title,
                revenue_cents AS revenue_cents,
                units AS units
            ORDER BY revenue_cents DESC
            LIMIT 200
        '''

    @override
    def _customer_spend_buckets(self, date):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
                AND o.ordered_at >= datetime('{date}')
            WITH p.person_id AS person_id, sum(o.total_cents) AS spend_cents
            WITH
                CASE
                    WHEN spend_cents < 5000 THEN 'low'
                    WHEN spend_cents < 20000 THEN 'mid'
                    ELSE 'high'
                END AS bucket
            RETURN bucket, count(*) AS persons
            ORDER BY bucket
        '''

    @override
    def _fraud_pattern(self, date, distinct_sellers_threshold):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
                AND o.ordered_at >= datetime('{date}')
            MATCH (o)-[:HAS_ITEM]->(pr:Product)
            MATCH (s:Seller)-[:OFFERS]->(pr)
            WITH p.person_id AS person_id,
                collect(DISTINCT s.seller_id) AS seller_ids,
                collect(DISTINCT o.order_id) AS order_ids
            WITH person_id,
                size(seller_ids) AS distinct_sellers,
                size(order_ids) AS orders
            WHERE distinct_sellers >= {distinct_sellers_threshold}
            RETURN person_id, distinct_sellers, orders
            ORDER BY distinct_sellers DESC
            LIMIT 200
        '''

    @override
    def _who_to_follow(self, person_id):
        return f'''
            MATCH (p:Person {{person_id: {person_id}}})<-[:FOLLOWS]-(:Person)<-[:FOLLOWS]-(cand:Person)
            WHERE NOT (p)-[:FOLLOWS]->(cand) AND cand.person_id <> {person_id}
            RETURN cand.person_id AS person_id, COUNT(*) AS paths
            ORDER BY paths DESC
            LIMIT 200
        '''

    @override
    def _personalized_feed_candidates(self, person_ids):
        return f'''
            MATCH (p:Person)-[hi:HAS_INTEREST]->(c:Category)<-[:HAS_CATEGORY]-(pr:Product)
            WHERE
                p.person_id IN [{person_ids}]
                AND pr.is_active = true
            RETURN pr.product_id AS product_id, SUM(hi.strength) AS interest_score
            ORDER BY interest_score DESC
            LIMIT 200
        '''


    # Document focused (MongoDB)
    # These are built to avoid joins at read time. Put "product page bundle" in one document.

    @override
    def _product_page_read(self, product_id):
        return f'''
            MATCH (pr:Product {{product_id: {product_id}, is_active: true}})
            MATCH (s:Seller)-[:OFFERS]->(pr)
            OPTIONAL MATCH (c:Customer)-[r:REVIEWED]->(pr)
            WITH pr, s, r, c
            ORDER BY r.helpful_votes DESC, r.created_at DESC
            WITH
                pr,
                s,
                avg(toFloat(r.rating)) AS review_avg,
                count(r) AS review_count,
                collect({{
                    customer_id: c.customer_id,
                    rating: r.rating,
                    title: r.title,
                    body: r.body,
                    created_at: r.created_at,
                    helpful_votes: r.helpful_votes
                }})[0..50] AS top_reviews
            RETURN
                pr.product_id AS product_id,
                pr.title AS title,
                pr.price_cents AS price_cents,
                pr.currency AS currency,
                pr.stock_qty AS stock_qty,
                {{
                    seller_id: s.seller_id,
                    display_name: s.display_name
                }} AS seller,
                coalesce(review_avg, 0) AS review_avg,
                review_count,
                top_reviews;
        '''

    # Bulk fetch product pages for a feed (Mongo, OLTP read-heavy, high weight)
    # def _bulk_fetch_product_pages(self):
    #     return f'''
    #         MATCH (pr:Product)
    #         WHERE pr.product_id IN [{self._param_product_ids(5, 20)}]
    #             AND pr.is_active = true
    #         MATCH (s:Seller)-[:OFFERS]->(pr)
    #         OPTIONAL MATCH (:Customer)-[r:REVIEWED]->(pr)
    #         WITH pr, s, avg(toFloat(r.rating)) AS avg_rating, count(r) AS review_count
    #         RETURN
    #             pr.product_id AS product_id,
    #             pr.title AS title,
    #             pr.price_cents AS price_cents,
    #             pr.currency AS currency,
    #             {{seller_id: s.seller_id, display_name: s.display_name}} AS seller,
    #             {{avg: COALESCE(avg_rating, 0.0), count: review_count}} AS rating_summary
    #         LIMIT 200
    #     '''

    # Graph focused (Neo4j)

    @override
    def _people_also_bought(self, product_ids):
        return f'''
            MATCH (target:Product)<-[:HAS_ITEM]-(o:Order)-[:HAS_ITEM]->(other:Product)
            WHERE target.product_id IN [{product_ids}]
                AND NOT other.product_id IN [{product_ids}]
                AND o.status IN ['paid', 'shipped']
            RETURN
                other.product_id AS product_id,
                COUNT(*) AS co_buy
            ORDER BY co_buy DESC
            LIMIT 20
        '''

    @override
    def _order_revenue_by_status_currency(self, statuses):
        return f'''
            MATCH (o:Order)
            WHERE o.status IN [{statuses}]
            RETURN
                o.status AS status,
                o.currency AS currency,
                count(*) AS orders,
                sum(o.total_cents) AS revenue_cents,
                avg(toFloat(o.total_cents)) AS avg_order_cents
            ORDER BY status, currency
        '''

    @override
    def _product_inventory_by_price_band(self, max_price_cents, min_stock_qty):
        return f'''
            MATCH (p:Product)
            WHERE p.is_active = true
                AND p.price_cents <= {max_price_cents}
                AND p.stock_qty >= {min_stock_qty}
            WITH
                CASE
                    WHEN p.price_cents < 1000 THEN 0
                    WHEN p.price_cents < 5000 THEN 1
                    WHEN p.price_cents < 20000 THEN 2
                    ELSE 3
                END AS price_band,
                p.currency AS currency,
                p.stock_qty AS stock_qty,
                p.price_cents AS price_cents
            RETURN
                price_band,
                currency,
                count(*) AS products,
                sum(stock_qty) AS stock_qty,
                avg(toFloat(price_cents)) AS avg_price_cents
            ORDER BY currency, price_band
        '''

    @override
    def _review_rating_distribution(self, min_helpful_votes):
        return f'''
            MATCH ()-[r:REVIEWED]->()
            WHERE r.helpful_votes >= {min_helpful_votes}
            RETURN
                r.rating AS rating,
                count(*) AS reviews,
                avg(toFloat(r.helpful_votes)) AS avg_helpful_votes
            ORDER BY rating
        '''

    @override
    def _customer_snapshot_activity(self, country_codes):
        return f'''
            MATCH (c:Customer)
            WHERE c.country_code IN [{country_codes}]
            RETURN
                c.country_code AS country_code,
                c.is_active AS is_active,
                count(*) AS customers,
                count(DISTINCT c.person_id) AS persons
            ORDER BY country_code, is_active
        '''

    @override
    def _seller_activity_rollup_by_country(self, country_codes):
        return f'''
            MATCH (s:Seller)
            WHERE s.country_code IN [{country_codes}]
            RETURN
                s.country_code AS country_code,
                s.is_active AS is_active,
                count(*) AS sellers
            ORDER BY country_code, is_active
        '''

    @override
    def _line_item_quantity_distribution(self, min_unit_price_cents):
        return f'''
            MATCH ()-[it:HAS_ITEM]->()
            WHERE it.unit_price_cents >= {min_unit_price_cents}
            RETURN
                it.quantity AS quantity,
                count(*) AS items,
                sum(it.quantity) AS units,
                sum(it.line_total_cents) AS revenue_cents,
                avg(toFloat(it.unit_price_cents)) AS avg_unit_price_cents
            ORDER BY quantity
        '''

    @override
    def _seller_sales_summary(self, date, seller_ids):
        return f'''
            MATCH (s:Seller)-[:OFFERS]->(pr:Product)<-[it:HAS_ITEM]-(o:Order)
            WHERE s.seller_id IN [{seller_ids}]
                AND o.ordered_at >= datetime('{date}')
                AND o.status IN ['paid', 'shipped']
            RETURN
                s.seller_id AS seller_id,
                count(DISTINCT o.order_id) AS orders,
                count(*) AS items,
                sum(it.quantity) AS units,
                sum(it.line_total_cents) AS revenue_cents
            ORDER BY revenue_cents DESC, seller_id
            LIMIT 200
        '''

    @override
    def _customer_country_order_status(self, date, country_codes):
        return f'''
            MATCH (c:Customer)-[:PLACED]->(o:Order)
            WHERE c.country_code IN [{country_codes}]
                AND o.ordered_at >= datetime('{date}')
            RETURN
                c.country_code AS country_code,
                o.status AS status,
                count(*) AS orders,
                sum(o.total_cents) AS revenue_cents,
                avg(toFloat(o.total_cents)) AS avg_order_cents
            ORDER BY country_code, status
        '''

    @override
    def _product_review_summary(self, product_ids, min_helpful_votes):
        return f'''
            MATCH ()-[r:REVIEWED]->(p:Product)
            WHERE p.product_id IN [{product_ids}]
                AND r.helpful_votes >= {min_helpful_votes}
            RETURN
                p.product_id AS product_id,
                count(*) AS reviews,
                avg(toFloat(r.rating)) AS avg_rating,
                sum(r.helpful_votes) AS helpful_votes,
                max(r.helpful_votes) AS max_helpful_votes
            ORDER BY reviews DESC, product_id
            LIMIT 200
        '''

    @override
    def _category_interest_summary(self, category_ids, min_strength):
        return f'''
            MATCH (p:Person)-[hi:HAS_INTEREST]->(c:Category)
            WHERE c.category_id IN [{category_ids}]
                AND hi.strength >= {min_strength}
                AND p.is_active = true
            RETURN
                c.category_id AS category_id,
                count(DISTINCT p.person_id) AS interested_persons,
                avg(toFloat(hi.strength)) AS avg_strength,
                max(hi.strength) AS max_strength
            ORDER BY interested_persons DESC, category_id
            LIMIT 200
        '''

    @override
    def _category_catalog_summary(self, category_ids, max_price_cents, min_stock_qty):
        return f'''
            MATCH (p:Product)-[:HAS_CATEGORY]->(c:Category)
            WHERE c.category_id IN [{category_ids}]
                AND p.is_active = true
                AND p.price_cents <= {max_price_cents}
                AND p.stock_qty >= {min_stock_qty}
            RETURN
                c.category_id AS category_id,
                count(DISTINCT p.product_id) AS products,
                sum(p.stock_qty) AS stock_qty,
                avg(toFloat(p.price_cents)) AS avg_price_cents
            ORDER BY products DESC, category_id
            LIMIT 200
        '''

    @override
    def _seller_catalog_health(self, seller_ids):
        return f'''
            MATCH (s:Seller)-[:OFFERS]->(p:Product)
            WHERE s.seller_id IN [{seller_ids}]
            RETURN
                s.seller_id AS seller_id,
                count(*) AS products,
                sum(CASE WHEN p.is_active THEN 1 ELSE 0 END) AS active_products,
                sum(CASE WHEN p.is_active THEN 0 ELSE 1 END) AS inactive_products,
                sum(p.stock_qty) AS stock_qty,
                avg(toFloat(p.price_cents)) AS avg_price_cents
            ORDER BY products DESC, seller_id
            LIMIT 200
        '''

    @override
    def _follow_country_rollup(self, country_codes):
        return f'''
            MATCH (p_from:Person)-[:FOLLOWS]->(p_to:Person)
            WHERE p_from.country_code IN [{country_codes}]
            RETURN
                p_from.country_code AS from_country,
                p_to.country_code AS to_country,
                count(*) AS edges,
                sum(CASE WHEN p_to.is_active THEN 1 ELSE 0 END) AS active_targets
            ORDER BY edges DESC, from_country, to_country
            LIMIT 200
        '''
