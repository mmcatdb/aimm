from abc import ABC, abstractmethod
import json
import os
from core.query import SchemaId
from core.files import JsonEncoder, open_input, open_output

class BaseLoader(ABC):
    """Base class for data loaders."""

    def _reset(self, driver, schema_id: SchemaId, import_directory: str):
        self._driver = driver
        self._schema_id = schema_id
        self._import_directory = import_directory
        self._times = dict[str, float]()

    @abstractmethod
    def run(self, driver, schema_id: SchemaId, import_directory: str, do_reset: bool) -> dict[str, float]:
        """Loads data from the specified directory into the database.

        If `do_reset` is True, the database will be cleared beforehand.
        Returns a dictionary mapping entity kinds to loading times in milliseconds.
        """
    # The driver is Any because it's overriden by the specific loaders. There is surely a better way to do this (probably with generics), but it works for now.

def save_populate_times(path: str, times: dict[str, float]):
    with open_output(path) as file:
        json.dump(times, file, cls=JsonEncoder, indent=4)

def load_populate_times(path: str) -> dict[str, float]:
    if not os.path.isfile(path):
        raise Exception(f'Populate times file not found: {path}')

    with open_input(path) as file:
        return json.load(file)
