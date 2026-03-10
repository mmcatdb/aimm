from common.database import Database

available_databases = [
    'edbt_neo4j',
    'edbt_postgres',
    'tpch_neo4j',
    'tpch_postgres',
]

def get_database_by_id(id: str) -> Database:
    if id == 'edbt_neo4j':
        from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
        return EdbtNeo4jDatabase()
    elif id == 'edbt_postgres':
        from datasets.edbt.postgres_database import EdbtPostgresDatabase
        return EdbtPostgresDatabase()
    elif id == 'tpch_neo4j':
        from datasets.tpch.neo4j_database import TpchNeo4jDatabase
        return TpchNeo4jDatabase()
    elif id == 'tpch_postgres':
        from datasets.tpch.postgres_database import TpchPostgresDatabase
        return TpchPostgresDatabase()
    else:
        raise ValueError(f'Database with ID: {id} not found. Available IDs: {available_databases}')
