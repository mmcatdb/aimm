from datasets.edbt.postgres_database import EdbtPostgresDatabase
from latency_estimation.postgres.context import PostgresContext
from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
from latency_estimation.neo4j.context import Neo4jContext
from common.drivers import cypher
import pprint

def main():
    ctx = PostgresContext.create(database=EdbtPostgresDatabase())
    # ctx = Neo4jContext.create(database=EdbtNeo4jDatabase())

    query = '''
        SELECT product.title
        FROM product
        JOIN seller ON product.seller_id = seller.seller_id
        -- WHERE product.title = 'abc'
        WHERE product_id > $1 AND product_id < $2
        LIMIT 10
    '''

    # query = '''
    #     MATCH (p:Person {{person_id: {person_id}}})<-[:SNAPSHOT_OF]-(c:Customer)-[:PLACED]->(o:Order)
    #     RETURN
    #         o.order_id AS order_id,
    #         o.ordered_at AS ordered_at,
    #         o.status AS status,
    #         o.total_cents AS total_cents,
    #         o.currency AS currency
    #     ORDER BY o.ordered_at DESC
    #     LIMIT 20
    #     '''.format(person_id=1)

    # plan = ctx.extractor.explain_plan(query)

    with ctx.driver.cursor() as cursor:
        cursor.execute(f'EXPLAIN (GENERIC_PLAN) {query}')
        result = cursor.fetchall()
        assert result is not None, 'No plan returned from EXPLAIN.'
        plan = '\n'.join([row[0] for row in result])

    # with ctx.driver.session() as session:
    #     result = session.run(cypher(f'EXPLAIN {query}'))
    #     plan = result.consume().plan
    #     assert plan is not None, 'Failed to retrieve query plan.'

    print(plan)
    # pprint.pprint(plan)

if __name__ == '__main__':
    main()
