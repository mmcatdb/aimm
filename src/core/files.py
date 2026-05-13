from collections.abc import Iterator
from io import TextIOWrapper
import os
from typing import Any
import dataclasses
import json
from bson import json_util

def open_output(path: str, skip_dir_check: bool = False) -> TextIOWrapper:
    if not skip_dir_check:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, 'w', newline='', encoding='utf-8')

def open_input(path: str) -> TextIOWrapper:
    return open(path, 'r', newline='', encoding='utf-8')

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            dataclass: Any = o
            return dataclasses.asdict(dataclass)
        return super().default(o)

class JsonLinesWriter:
    def __init__(self, file: TextIOWrapper, extended: bool):
        """If `extended` is True, extended json (via `bson.json_util`) will be used."""
        self._file = file
        self._extended = extended

    def writeobject(self, object: dict):
        if self._extended:
            self._file.write(json_util.dumps(object))
        else:
            json.dump(object, self._file, cls=JsonEncoder)
        self._file.write('\n')

class JsonLinesReader:
    def __init__(self, file: TextIOWrapper, extended: bool):
        """If `extended` is True, extended json (via `bson.json_util`) will be used."""
        self._file = file
        self._extended = extended

    def readobject(self) -> dict:
        line = self._file.readline()
        if not line:
            raise EOFError()
        return self._parse(line)

    def readobjects(self) -> list[dict]:
        return [self._parse(line) for line in self._file]

    def __iter__(self) -> Iterator[dict]:
        for line in self._file:
            yield self._parse(line)

    def _parse(self, line: str) -> dict:
        return json_util.loads(line) if self._extended else json.loads(line)
