from typing import Generator, cast
from contextlib import contextmanager
from typing_extensions import LiteralString
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import connection as PostgresConnection
from psycopg2.extensions import cursor as PostgresCursor
from pymongo import MongoClient
from neo4j import GraphDatabase

# These classes manage database configurations and connections.
# They should be created one per application and reused.

class PostgresConfig:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

class Postgres():
    """
    Usage:
    with postgres.cursor() as cursor:
        cursor.execute('SELECT 1')
    """
    def __init__(self, config: PostgresConfig):
        self._config = config
        self._pool = SimpleConnectionPool(
            minconn = 1,
            maxconn = 10,
            host = config.host,
            port = config.port,
            database = config.database,
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
        except Exception:
            connection.rollback()
            raise
        finally:
            self._pool.putconn(connection)

    def get_connection(self) -> PostgresConnection:
        """Useful when you need to manage transactions manually. Don't forget to put it back to the pool!"""
        return self._pool.getconn()

    def put_connection(self, connection: PostgresConnection):
        self._pool.putconn(connection)

class MongoConfig:
    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database

class Mongo():
    """
    Usage:
    database = mongo.database()
    database.myCollection.find({ ... })
    """
    def __init__(self, config: MongoConfig):
        self._config = config
        self._client = MongoClient(f'mongodb://{config.host}:{config.port}')
        self._database = self._client[config.database]

    def database(self):
        return self._database

class Neo4jConfig:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

class Neo4j():
    """
    Usage:
    with neo4j.session() as session:
        result = session.run('MATCH (n) RETURN n LIMIT 1')
    """
    def __init__(self, config: Neo4jConfig):
        self._config = config
        self._driver = GraphDatabase.driver(
            f'bolt://{config.host}:{config.port}',
            auth = (config.user, config.password)
        )

    def session(self):
        """No need to specify database since Neo4j doesn't support multiple databases in community edition (they should be deeply ashamed)."""
        return self._driver.session()

    def verify(self):
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

def cypher(query: str) -> LiteralString:
    """Nobody asked for this feature. What about some actually useful types hints instead?"""
    return cast(LiteralString, query)
