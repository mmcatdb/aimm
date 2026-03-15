from common.config import DatasetName
from latency_estimation.postgres.context import PostgresContext
from latency_estimation.neo4j.context import Neo4jContext
from common.drivers import cypher
import pprint
from experiments.__main__ import main as experiments_main

# FIXME this
TEST_DATASET = DatasetName.EDBT

def main():
    # test_plans()
    test_evaluation()

def test_evaluation():
    # experiments_main(split('evaluate -c data/checkpoints/tpch_neo4j_final.pt -d neo4j'))

    experiments_main(split('evaluate -c data/checkpoints/tpch_postgres_final.pt -d postgres'))

def split(args: str) -> list[str]:
    """Unfortunately, python treats output of string.split as list[LiteralString], which is not compatible with list[str]."""
    return args.split(' ')

def test_plans():
    ctx = PostgresContext.create(dataset=TEST_DATASET)
    # ctx = Neo4jContext.create(database=TEST_DATASET)

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
