from core.config import Config
from core.driver_provider import DriverProvider
from search.schemas.tpch.query_engine import TpchQueryEngine
from search.mcts_old import MCTS

def main():
    tables = ('customer', 'orders', 'supplier', 'part', 'partsupp', 'lineitem')
    initial_mapping = tuple(['postgres'] * len(tables))
    dbs = DriverProvider.default(Config.load())
    query_engine = TpchQueryEngine.create(dbs)
    mcts = MCTS(query_engine, tables)
    best_mapping, best_time = mcts.run(initial_mapping, iterations=50)
    print(f'Final best mapping: {best_mapping} with time {best_time}s')

if __name__ == '__main__':
    main()
