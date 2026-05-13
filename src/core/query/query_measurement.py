from __future__ import annotations
from dataclasses import dataclass
from io import TextIOWrapper
import os
from typing import Generic, cast
from core.files import JsonLinesReader, JsonLinesWriter, open_input, open_output
from .query_id import DatabaseId, QueryInstanceId, parse_query_instance_driver_type
from .query_instance import QueryInstance, TQuery, parse_query

class QueryMeasurement(QueryInstance[TQuery]):
    """All measured data about a `QueryInstance`."""

    def __init__(self, id: QueryInstanceId, label: str, is_write: bool, content: TQuery, plan: dict, times: list[float]):
        super().__init__(id, label, is_write, content)
        self.plan = plan
        self.times = times

    @staticmethod
    def from_instance(instance: QueryInstance[TQuery], plan: dict, times: list[float]) -> QueryMeasurement[TQuery]:
        return QueryMeasurement(
            id=instance.id,
            label=instance.label,
            is_write=instance.is_write,
            content=instance.content,
            plan=plan,
            times=times
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'is_write': self.is_write,
            'content': serialize_query(self.content),
            'plan': self.plan,
            'times': self.times,
        }

    @staticmethod
    def from_dict(data: dict) -> QueryMeasurement:
        id = data['id']
        driver_type = parse_query_instance_driver_type(id)
        content = parse_query(driver_type, data['content'])
        return QueryMeasurement(
            id=id,
            label=data['label'],
            is_write=data['is_write'],
            # This is probably necessary due to the limitations of python's type system.
            content=cast(TQuery, content),
            plan=data['plan'],
            times=data['times']
        )

def serialize_query(query: TQuery) -> str:
    if isinstance(query, str):
        return query
    return query.serialize()

@dataclass
class MeasurementConfig:
    num_queries: int
    num_runs: int
    allow_write: bool

class MeasuredQueries(Generic[TQuery]):
    """Represents a collection of measured queries, including their execution plans and latencies, as well as some global statistics about the database.

    Also contains some metadata for debugging.
    """

    def __init__(self, items: list[QueryMeasurement[TQuery]], global_stats: dict, database_id: DatabaseId, config: MeasurementConfig):
        self.items = items
        self.global_stats = global_stats
        self.database_id = database_id
        self.num_queries = config.num_queries
        self.num_runs = config.num_runs
        self.allow_write = config.allow_write

    def is_fully_measured(self) -> bool:
        return len(self.items) == self.num_queries

class MeasuredQueriesPersistor(Generic[TQuery]):

    def __init__(self, file: TextIOWrapper, writer: JsonLinesWriter) -> None:
        self._file = file
        self._writer = writer

    @staticmethod
    def open(path: str) -> MeasuredQueriesPersistor[TQuery]:
        file = open(path, 'a', newline='', encoding='utf-8')
        writer = JsonLinesWriter(file, extended=True)
        return MeasuredQueriesPersistor(file, writer)

    def append(self, measurement: QueryMeasurement[TQuery]):
        self._writer.writeobject(measurement.to_dict())

    def close(self):
        self._file.close()

def save_measured(path: str, measured: MeasuredQueries):
    with open_output(path) as file:
        writer = JsonLinesWriter(file, extended=True)

        # Let's use json lines for better readability.
        writer.writeobject({
            'database_id': measured.database_id,
            'num_queries': measured.num_queries,
            'num_runs': measured.num_runs,
            'allow_write': measured.allow_write,
            'global_stats': measured.global_stats,
        })

        for item in measured.items:
            writer.writeobject(item.to_dict())

def load_measured(path: str) -> MeasuredQueries:
    with open_input(path) as file:
        reader = JsonLinesReader(file, extended=True)
        header = reader.readobject()
        items = [QueryMeasurement.from_dict(item) for item in reader]
        mc = MeasurementConfig(
            num_queries=int(header['num_queries']),
            num_runs=int(header['num_runs']),
            allow_write=bool(header['allow_write'])
        )

        return MeasuredQueries(items, header['global_stats'], header['database_id'], mc)
