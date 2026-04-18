from __future__ import annotations
import json
import os
from typing import Generic, cast
from core.utils import JsonEncoder
from .query_id import DatabaseId, QueryInstanceId, parse_query_instance_driver_type
from .query_instance import TQuery, parse_query

class QueryMeasurement(Generic[TQuery]):
    """All measured data about a `QueryInstance`."""

    def __init__(self, query_id: QueryInstanceId, content: TQuery, plan: dict, times: list[float]):
        self.query_id = query_id
        self.content = content
        self.plan = plan
        self.times = times

    def to_dict(self) -> dict:
        return {
            'query_id': self.query_id,
            'content': serialize_query(self.content),
            'plan': self.plan,
            'times': self.times,
        }

    @staticmethod
    def from_dict(data: dict) -> QueryMeasurement:
        query_id = data['query_id']
        driver_type = parse_query_instance_driver_type(query_id)
        content = parse_query(driver_type, data['content'])
        return QueryMeasurement(
            query_id=query_id,
            # This is probably necessary due to the limitations of python's type system.
            content=cast(TQuery, content),
            plan=data['plan'],
            times=data['times']
        )

def serialize_query(query: TQuery) -> str:
    if isinstance(query, str):
        return query
    return query.serialize()

class MeasuredQueries(Generic[TQuery]):
    """Represents a collection of measured queries, including their execution plans and latencies, as well as some global statistics about the database.

    Also contains some metadata for debugging.
    """

    def __init__(self, items: list[QueryMeasurement[TQuery]], global_stats: dict, database_id: DatabaseId, num_queries: int, num_runs: int):
        self.items = items
        self.global_stats = global_stats
        self.database_id = database_id
        self.num_queries = num_queries
        self.num_runs = num_runs

def save_measured(path: str, measured: MeasuredQueries):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as file:
        # Let's use json lines for better readability.
        json.dump({
            'database_id': measured.database_id,
            'num_queries': measured.num_queries,
            'num_runs': measured.num_runs,
            'global_stats': measured.global_stats,
        }, file, cls=JsonEncoder)
        file.write('\n')

        for item in measured.items:
            json.dump(item.to_dict(), file, cls=JsonEncoder)
            file.write('\n')

def load_measured(path: str) -> MeasuredQueries:
    with open(path, 'r') as file:
        header = json.loads(file.readline())
        items = [QueryMeasurement.from_dict(json.loads(line)) for line in file]

        return MeasuredQueries(
            items,
            header['global_stats'],
            header['database_id'],
            int(header['num_queries']),
            int(header['num_runs']),
        )
