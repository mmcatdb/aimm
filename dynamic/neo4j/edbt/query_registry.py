from core.query import query
from core.drivers import DriverType
from ...common.edbt.query_registry import EdbtQueryRegistry

def export():
    return Neo4jEdbtQueryRegistry()

class Neo4jEdbtQueryRegistry(EdbtQueryRegistry[str]):

    def __init__(self):
        super().__init__(DriverType.NEO4J)

    # OLTP focused (mostly Postgres)

    @query('edbt-0', 'Order history for a person (order, customer)')
    def _order_history_for_person(self):
        return f'''
            MATCH (p:Person {{person_id: {self._param_person_id()}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            RETURN
                o.order_id AS order_id,
                o.ordered_at AS ordered_at,
                o.status AS status,
                o.total_cents AS total_cents,
                o.currency AS currency
            ORDER BY o.ordered_at DESC
        '''

    @query('edbt-1', 'Order details view (order, customer, order_item, product)')
    def _order_details(self):
        return f'''
            MATCH (p:Person {{person_id: {self._param_person_id()}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
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

    @query('edbt-2', 'How many times did this person bought these products? (order, customer, order_item)')
    def _product_purchases_for_person(self):
        person_ids = self._param_person_ids(100, 1000)
        product_ids = self._param_product_ids(100, 1000)

        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
            MATCH (o)-[:HAS_ITEM]->(pr:Product)
            WHERE p.person_id IN [{person_ids}]
                AND pr.product_id IN [{product_ids}]
            RETURN COUNT(*) AS purchase_count
        '''

    # OLAP focused (Postgres)

    @query('edbt-3', 'Seller daily revenue for last 30 days (order, order_item, product)')
    def _seller_daily_revenue(self):
        date = self._param_date_minus_days(30, 120)
        seller_ids = self._param_seller_ids(100, 1000)

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

    @query('edbt-4', 'Top products by revenue inside one category, last 7-30 days (order, order_item, product, has_category)')
    def _top_products_by_revenue(self):
        date = self._param_date_minus_days(7, 30)
        category_ids = self._param_category_ids(10, 50)

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

    @query('edbt-5', 'Customer spend buckets (order, customer)')
    def _customer_spend_buckets(self):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
                AND o.ordered_at >= datetime('{self._param_date_minus_days(30, 180)}')
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

    @query('edbt-6', 'Fraud-ish pattern (order, customer, order_item, product)')
    def _fraud_pattern(self):
        date = self._param_date_minus_days(1, 7)
        distinct_sellers_threshold = self._param_int('distinct_sellers_threshold', 10, 1000)

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

    @query('edbt-7', 'Who should I follow? (follows)')
    def _who_to_follow(self):
        person_id = self._param_person_id()

        return f'''
            MATCH (p:Person {{person_id: {person_id}}})<-[:FOLLOWS]-(:Person)<-[:FOLLOWS]-(cand:Person)
            WHERE NOT (p)-[:FOLLOWS]->(cand) AND cand.person_id <> {person_id}
            RETURN cand.person_id AS person_id, COUNT(*) AS paths
            ORDER BY paths DESC
            LIMIT 200
        '''

    @query('edbt-8', 'Personalized feed candidates (product, has_category, has_interest)')
    def _personalized_feed_candidates(self):
        return f'''
            MATCH (p:Person)-[hi:HAS_INTEREST]->(c:Category)<-[:HAS_CATEGORY]-(pr:Product)
            WHERE
                p.person_id IN [{self._param_person_ids(2, 20)}]
                AND pr.is_active = true
            RETURN pr.product_id AS product_id, SUM(hi.strength) AS interest_score
            ORDER BY interest_score DESC
            LIMIT 200
        '''


    # Document focused (MongoDB)
    # These are built to avoid joins at read time. Put "product page bundle" in one document.

    @query('edbt-9', 'Product page read (product, seller, review)')
    def _product_page_read(self):
        return f'''
            MATCH (pr:Product {{product_id: {self._param_product_id()}, is_active: true}})
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



    @query('edbt-10', 'People also bought using shared orders (order, order_item)')
    def _people_also_bought(self):
        product_ids = self._param_product_ids(10, 50)

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

