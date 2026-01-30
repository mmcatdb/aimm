import yaml
import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseConfig:
    """Manages database configuration and connections."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.user = config['postgres']['user']
        self.password = config['postgres']['password']
        self.host = config['postgres']['host']
        self.port = config['postgres']['port']
        self.dbname = config['postgres']['dbname']

    def get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            dbname=self.dbname
        )

    def execute_query(self, query: str, fetch: bool = True):
        """Execute a query and optionally fetch results."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
        finally:
            conn.close()
