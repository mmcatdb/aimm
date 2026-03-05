"""
MongoDB configuration and connection management.
"""
import pymongo
from typing import Dict, Any, Optional


class MongoConfig:
    """Manages MongoDB configuration and connections."""

    def __init__(self, host: str = "localhost", port: int = 27017,
                 dbname: str = "tpch"):
        self.host = host
        self.port = port
        self.dbname = dbname
        self._client = None

    def get_client(self) -> pymongo.MongoClient:
        """Get or create a MongoClient."""
        if self._client is None:
            self._client = pymongo.MongoClient(self.host, self.port)
        return self._client

    def get_db(self):
        """Get database handle."""
        return self.get_client()[self.dbname]

    def get_collection(self, name: str):
        """Get collection handle."""
        return self.get_db()[name]

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics via collStats command."""
        db = self.get_db()
        stats = db.command("collStats", collection_name)
        return {
            "count": stats.get("count", 0),
            "size": stats.get("size", 0),
            "avgObjSize": stats.get("avgObjSize", 0),
            "storageSize": stats.get("storageSize", 0),
            "nindexes": stats.get("nindexes", 0),
            "totalIndexSize": stats.get("totalIndexSize", 0),
        }

    def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        db = self.get_db()
        stats = {}
        for name in db.list_collection_names():
            if not name.startswith("system."):
                stats[name] = self.get_collection_stats(name)
        return stats

    def explain_find(self, collection_name: str, filter_doc: dict,
                     projection: Optional[dict] = None,
                     sort: Optional[dict] = None,
                     limit: int = 0, skip: int = 0,
                     verbosity: str = "queryPlanner") -> Dict:
        """
        Run explain on a find command.

        Args:
            verbosity: 'queryPlanner' (no execution) or 'executionStats' (runs query)
        """
        db = self.get_db()
        cmd = {"find": collection_name, "filter": filter_doc}
        if projection:
            cmd["projection"] = projection
        if sort:
            cmd["sort"] = sort
        if limit:
            cmd["limit"] = limit
        if skip:
            cmd["skip"] = skip
        return db.command("explain", cmd, verbosity=verbosity)

    def explain_aggregate(self, collection_name: str, pipeline: list,
                          verbosity: str = "queryPlanner") -> Dict:
        """Run explain on an aggregate pipeline."""
        db = self.get_db()
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        return db.command("explain", cmd, verbosity=verbosity)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
