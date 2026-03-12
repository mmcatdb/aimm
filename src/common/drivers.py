from collections.abc import Generator
from enum import Enum
from typing import cast
from contextlib import contextmanager
from typing_extensions import LiteralString
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import connection as PostgresConnection
from psycopg2.extensions import cursor as PostgresCursor
from pymongo import MongoClient
from neo4j import GraphDatabase

class DriverType(Enum):
    POSTGRES = 'postgres'
    MONGO = 'mongo'
    NEO4J = 'neo4j'

# These classes manage database configurations and connections.
# They should be created once per application and reused.

class PostgresConfig:
    def __init__(self, host: str, port: int, root_database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.root_database = root_database
        self.user = user
        self.password = password

class PostgresDriver():
    """
    Usage:
    with postgres.cursor() as cursor:
        cursor.execute('SELECT 1')
    """
    def __init__(self, config: PostgresConfig, database: str):
        self.config = config
        self.database = database
        # TODO Not ideal, this tries to connect immediately. We might want to defer it.
        self._pool = SimpleConnectionPool(
            minconn = 1,
            maxconn = 10,
            host = config.host,
            port = config.port,
            database = database,
            user = config.user,
            password = config.password,
        )

    @contextmanager
    def cursor(self, *args, **kwargs) -> Generator[PostgresCursor, None, None]:
        """Most high-level method. Doesn't expose the connection."""
        with self.connection() as connection:
            with connection.cursor(*args, **kwargs) as cursor:
                yield cursor

    @contextmanager
    def connection(self) -> Generator[PostgresConnection, None, None]:
        """Borrows a raw connection from the pool. The transaction is automatically committed on success / rollbacked on error."""
        connection = self._pool.getconn()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise e
        finally:
            self._pool.putconn(connection)

    def get_connection(self) -> PostgresConnection:
        """Useful when you need to manage transactions manually. Don't forget to put it back to the pool!"""
        return self._pool.getconn()

    def put_connection(self, connection: PostgresConnection):
        self._pool.putconn(connection)

    def close(self):
        self._pool.closeall()

    def query_total_size(self) -> int:
        """Returns the total size of the database in bytes."""
        with self.cursor() as cursor:
            cursor.execute("""
                SELECT SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)))
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
            """)
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else 0

    def query_record_count(self) -> int:
        """Returns the total number of records in all tables."""
        with self.cursor() as cursor:
            cursor.execute("""
                SELECT SUM(reltuples::BIGINT)
                FROM pg_class
                WHERE relkind = 'r' AND relnamespace IN (
                    SELECT oid FROM pg_namespace WHERE nspname NOT IN ('pg_catalog', 'information_schema')
                );
            """)
            result = cursor.fetchone()
            return int(result[0]) if result and result[0] is not None else 0

class MongoConfig:
    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database

class MongoDriver():
    """
    Usage:
    database = mongo.database()
    database.myCollection.find({ ... })
    """
    def __init__(self, config: MongoConfig):
        self.config = config
        self._client = MongoClient(f'mongodb://{config.host}:{config.port}')
        self._database = self._client[config.database]

    def database(self):
        return self._database

    def close(self):
        self._client.close()

    def query_total_size(self) -> int:
        """Returns the total size of the database in bytes."""
        stats = self._database.command("dbstats")
        return stats.get("dataSize", 0) + stats.get("indexSize", 0)

    def query_record_count(self) -> int:
        """Returns the total number of documents in all collections."""
        total_count = 0
        for collection_name in self._database.list_collection_names():
            collection = self._database[collection_name]
            total_count += collection.estimated_document_count()
        return total_count

class Neo4jConfig:
    def __init__(self, host: str, ports: dict[str, int], user: str, password: str):
        self.host = host
        self.ports = ports
        self.user = user
        self.password = password

class Neo4jDriver():
    """
    Usage:
    with neo4j.session() as session:
        result = session.run('MATCH (n) RETURN n LIMIT 1')
    """
    def __init__(self, config: Neo4jConfig, database: str):
        self.config = config
        self.port = config.ports.get(database)

        # FIXME This shouldn't be in constructor, move it somewhere else.
        if self.port is None:
            raise ValueError(f'No port configured for database "{database}" in Neo4jConfig.')

        self._driver = GraphDatabase.driver(
            # FIXME Not pretty.
            f'bolt://{config.host}:{self.port}',
            auth = (config.user, config.password)
        )

    def session(self):
        """No need to specify database since Neo4j doesn't support multiple databases in community edition (they should be deeply ashamed)."""
        return self._driver.session()

    def verify(self):
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def query_record_count(self) -> int:
        """Returns the total number of nodes and relationships in the database."""
        with self.session() as session:
            result = session.run('MATCH (n) RETURN count(n) AS node_count').single()
            node_count = result.get('node_count', 0) if result else 0

            result = session.run('MATCH ()-[r]->() RETURN count(r) AS relationship_count').single()
            relationship_count = result.get('relationship_count', 0) if result else 0

            return node_count + relationship_count

    # Neo4j doesn't provide a straightforward way to get the total size of the database ... and we don't want to return inaccurate results.

def cypher(query: str) -> LiteralString:
    """Nobody asked for this feature. What about some actually useful types hints instead?"""
    return cast(LiteralString, query)
