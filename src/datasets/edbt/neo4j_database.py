from common.query_registry import query
from common.drivers import DriverType
from datasets.edbt.edbt_database import EdbtDatabase

class EdbtNeo4jDatabase(EdbtDatabase[str]):

    def __init__(self):
        super().__init__(DriverType.NEO4J)

    # OLTP focused (mostly Postgres)

    @query('test', 1.0, 'Order history for a person (join via customer)')
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
            LIMIT 20
        '''

    @query('test', 1.0, 'Order details view (now checks person via customer)')
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

    @query('test', 1.0, 'How many times did this person buy this product? (via customer snapshots)')
    def _product_purchases_for_person(self):
        return f'''
            MATCH (p:Person {{person_id: {self._param_person_id()}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
            MATCH (o)-[:HAS_ITEM]->(pr:Product {{product_id: {self._param_product_id()}}})
            RETURN COUNT(*) AS purchase_count
        '''

    # OLAP focused (Postgres)

    @query('test', 1.0, 'Seller daily revenue for last 30 days (Postgres, OLAP, medium weight)')
    def _seller_daily_revenue(self):
        return f'''
            MATCH (s:Seller {{seller_id: {self._param_seller_id()}}})-[:OFFERS]->(pr:Product)
            MATCH (o:Order)-[it:HAS_ITEM]->(pr)
            WHERE o.status IN ['paid', 'shipped']
              AND o.ordered_at >= datetime() - duration('P30D')
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

    @query('test', 1.0, 'Top products by revenue inside one category, last 7 days (Postgres, OLAP, high weight in sale)')
    def _top_products_by_revenue(self):
        return f'''
            MATCH (c:Category {{category_id: {self._param_category_id()}}})<-[:HAS_CATEGORY]-(pr:Product)
            MATCH (o:Order)-[it:HAS_ITEM]->(pr)
            WHERE o.status IN ['paid', 'shipped']
              AND o.ordered_at >= datetime() - duration('P7D')
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
            LIMIT 50
        '''

    @query('test', 1.0, 'Customer spend buckets (now per person)')
    def _customer_spend_buckets(self):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
              AND o.ordered_at >= datetime() - duration('P90D')
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

    @query('test', 1.0, 'Fraud-ish pattern (now per person)')
    def _fraud_pattern(self):
        return f'''
            MATCH (p:Person)<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
            WHERE o.status IN ['paid', 'shipped']
              AND o.ordered_at >= datetime() - duration('PT24H')
            MATCH (o)-[:HAS_ITEM]->(pr:Product)
            MATCH (s:Seller)-[:OFFERS]->(pr)
            WITH p.person_id AS person_id,
                collect(DISTINCT s.seller_id) AS seller_ids,
                collect(DISTINCT o.order_id) AS order_ids
            WITH person_id,
                size(seller_ids) AS distinct_sellers,
                size(order_ids) AS orders
            WHERE distinct_sellers >= 10
            RETURN person_id, distinct_sellers, orders
            ORDER BY distinct_sellers DESC
            LIMIT 200
        '''

    # Document focused (MongoDB)
    # These are built to avoid joins at read time. Put "product page bundle" in one document.

    @query('test', 1.0, 'Product page read (Mongo, OLTP read-heavy, very high weight in sale)')
    def _product_page_read(self):
        return f'''
            MATCH (pr:Product {{product_id: {self._param_product_id()}}})
            MATCH (s:Seller)-[:OFFERS]->(pr)
            OPTIONAL MATCH (c:Customer)-[r:REVIEWED]->(pr)
            RETURN
                pr.product_id AS product_id,
                pr.title AS title,
                pr.price_cents AS price_cents,
                pr.currency AS currency,
                pr.stock_qty AS stock_qty,
                {{seller_id: s.seller_id, display_name: s.display_name}} AS seller,
                avg(toFloat(r.rating)) AS review_avg,
                count(r) AS review_count,
                collect({{
                    customer_id: c.customer_id,
                    rating: r.rating,
                    title: r.title,
                    body: r.body,
                    created_at: r.created_at,
                    helpful_votes: r.helpful_votes
                }}) AS reviews
        '''

    @query('test', 1.0, 'Bulk fetch product pages for a feed (Mongo, OLTP read-heavy, high weight)')
    def _bulk_fetch_product_pages(self):
        return f'''
            MATCH (pr:Product)
            WHERE pr.product_id IN [{self._param_product_ids(5, 20)}]
              AND pr.is_active = true
            MATCH (s:Seller)-[:OFFERS]->(pr)
            OPTIONAL MATCH (:Customer)-[r:REVIEWED]->(pr)
            WITH pr, s, avg(toFloat(r.rating)) AS avg_rating, count(r) AS review_count
            RETURN
                pr.product_id AS product_id,
                pr.title AS title,
                pr.price_cents AS price_cents,
                pr.currency AS currency,
                {{seller_id: s.seller_id, display_name: s.display_name}} AS seller,
                {{avg: COALESCE(avg_rating, 0.0), count: review_count}} AS rating_summary
            LIMIT 200
        '''

    # Graph focused (Neo4j)

    @query('test', 1.0, 'Who should I follow? (User -> Person)')
    def _who_to_follow(self):
        person_id = self._param_person_id()

        return f'''
            MATCH (p:Person {{person_id: {person_id}}})-[:FOLLOWS]->(:Person)-[:FOLLOWS]->(cand:Person)
            WHERE NOT (p)-[:FOLLOWS]->(cand) AND cand.person_id <> {person_id}
            RETURN cand.person_id AS person_id, COUNT(*) AS paths
            ORDER BY paths DESC
            LIMIT 50
        '''

    @query('test', 1.0, 'Neo4j replacement for old "SIMILAR" query (since similar is removed) "People also bought" using shared orders')
    def _people_also_bought(self):
        product_id = self._param_product_id()

        return f'''
            MATCH (target:Product {{product_id: {product_id}}})<-[:HAS_ITEM]-(o:Order)-[:HAS_ITEM]->(other:Product)
            WHERE other.product_id <> {product_id}
            RETURN other.product_id AS product_id, COUNT(*) AS co_buy
            ORDER BY co_buy DESC
            LIMIT 20
        '''

    @query('test', 1.0, 'Personalized feed candidates (User -> Product)')
    def _personalized_feed_candidates(self):
        return f'''
            MATCH (p:Person {{person_id: {self._param_person_id()}}})-[hi:HAS_INTEREST]->(c:Category)<-[:HAS_CATEGORY]-(pr:Product)
            WHERE pr.is_active = true
            RETURN pr.product_id AS product_id, SUM(hi.strength) AS interest_score
            ORDER BY interest_score DESC
            LIMIT 200
        '''
