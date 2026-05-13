import csv
from abc import abstractmethod
from datetime import datetime
from enum import Enum
import json
import os
import re
import time
from pymongo import ASCENDING
from core.files import open_input
from core.query import JsonLinesReader, SchemaId
from core.utils import ProgressTracker, print_warning, time_quantity
from core.drivers import MongoDriver
from .base_loader import BaseLoader
from .postgres_loader import PostgresColumn

class MongoLoader(BaseLoader):
    """A class to load data into a Mongo database."""

    _driver: MongoDriver

    @abstractmethod
    def _get_csv_kinds(self) -> list['MongoDocument']:
        """Returns the schema for each csv entity kind. The order matters."""
        pass

    def _get_json_kinds(self) -> list[str]:
        """Returns collection names that should be loaded from json files."""
        return []

    @abstractmethod
    def _get_constraints(self) -> list['MongoIndex']:
        """Returns the list of constraints to be created."""
        pass

    def run(self, driver: MongoDriver, schema_id: SchemaId, import_directory: str, do_reset: bool):
        self._reset(driver, schema_id, import_directory)

        print(f'Loading data to Mongo at: {self._driver.config.host}:{self._driver.config.port}')

        self.__check_files()

        if do_reset:
            print('\nResetting database...')
            self.__reset_database()
            print('Database reset completed.')

        print('\nCreating constraints...')
        for index in self._get_constraints():
            self.__create_index(index)
        # TODO some other indexes?
        print('Constraints created.')

        print('\nLoading data...')
        for csv_kind in self._get_csv_kinds():
            self.__populate_csv_kind(csv_kind)
        for json_kind in self._get_json_kinds():
            self.__populate_json_kind(json_kind)
        print('Data loading completed.')

        return self._times

    def __check_files(self):
        """Verify that all files exist in the import directory."""
        kinds = set[str]()
        for kind in self._get_csv_kinds():
            kinds.update(self.__find_required_kinds(kind))

        filenames = [f'{kind}.tbl' for kind in kinds] + [f'{kind}.jsonl' for kind in self._get_json_kinds()]

        # Verify that all CSV files exist.
        for filename in filenames:
            path = os.path.join(self._import_directory, filename)
            if not os.path.isfile(path):
                raise Exception(f'Required file not found in import directory: {path}')

    def __find_required_kinds(self, kind: 'MongoDocument') -> set[str]:
        output = set[str]()
        output.add(kind.source.name)

        for property in kind.properties.values():
            if isinstance(property, ComplexProperty):
                output.update(self.__find_required_kinds(property.document))

        return output

    def __reset_database(self):
        database = self._driver.database()
        collection_names = database.list_collection_names()
        for entity in reversed(collection_names):
            try:
                database.drop_collection(entity)
                print(f'Collection "{entity}" has been dropped in MongoDB.')
            except Exception as e:
                print_warning(f'Skipping delete for "{entity}"', e)

    def __create_index(self, index: 'MongoIndex'):
        """The keys can be nested, e.g., `user.address.street`."""
        collection = self._driver.collection(index.kind)
        # The direction shouldn't matter. ASCENDING is the default.
        keys = [(key, ASCENDING) for key in index.keys]

        unique = 'unique ' if index.is_unique else ''
        message = f'{unique}index on "{index.keys}" for collection "{index.kind}"'

        try:
            collection.create_index(keys, unique=index.is_unique)
            print(f'Created {message}')
        except Exception as e:
            print_warning(f'Could not create {message}', e)

    def __populate_csv_kind(self, document: 'MongoDocument'):
        collection_name = document.source.name
        print(f'Loading collection "{collection_name}"... ')

        # We do this one collection at a time. This is somewhat wasteful, since we have to load some files multiple times, but it allows us to fit the whole process in memory.
        cache = self.__build_cache(document)
        rows = cache[document.source.name].get_all()
        documents = [self.__create_mongo_document(cache, document, row) for row in rows]

        self.__insert_documents(collection_name, documents)

    def __populate_json_kind(self, collection_name: str):
        print(f'Loading collection "{collection_name}"...')

        path = os.path.join(self._import_directory, collection_name + '.jsonl')
        with open_input(path) as file:
            reader = JsonLinesReader(file, extended=True)
            documents = list(reader)

        self.__insert_documents(collection_name, documents)

    def __insert_documents(self, collection_name: str, documents: list[dict]):
        collection = self._driver.collection(collection_name)
        start = time.perf_counter()
        print(f'Inserting {len(documents)} documents into collection "{collection_name}"... ', end='', flush=True)
        collection.insert_many(documents, ordered=False)
        self._times[collection_name] = time_quantity.to_base(time.perf_counter() - start, 's')
        print(time_quantity.pretty_print(self._times[collection_name]))

    def __build_cache(self, document: 'MongoDocument'):
        cache = dict[str, CachedCsvTable]()
        self.__find_kinds_to_cache(cache, document, None, None)

        # Load all kinds. The order doesn't matter, since they will reference each other via the cache, not via the database.
        for kind in cache.values():
            kind.load(self._import_directory)

        return cache

    def __find_kinds_to_cache(self, cache: dict[str, 'CachedCsvTable'], document: 'MongoDocument', index_by: 'CsvJoin | None', is_array: bool | None):
        table = cache.get(document.source.name)
        if not table:
            table = CachedCsvTable(document.source)
            cache[document.source.name] = table

        if index_by is not None:
            table.add_index(index_by, not is_array)

        for property in document.properties.values():
            if isinstance(property, ComplexProperty):
                self.__find_kinds_to_cache(cache, property.document, property.child_join, property.is_array)

    def __create_mongo_document(self, cache: dict[str, 'CachedCsvTable'], document: 'MongoDocument', row: 'CsvRow') -> dict:
        output = {}

        for key, property in document.properties.items():
            if isinstance(property, int):
                output[key] = row[property]
                continue

            child = property
            table = cache[child.document.source.name]

            parent_index_key = get_index_key_for_row(child.parent_join, row)

            if child.is_array:
                child_rows = table.get_non_unique(child.child_join, parent_index_key)
                child_value = [self.__create_mongo_document(cache, child.document, child_row) for child_row in child_rows]
            else:
                child_row = table.get_unique(child.child_join, parent_index_key)
                child_value = self.__create_mongo_document(cache, child.document, child_row)

            output[key] = child_value

        return output

class MongoIndex:
    def __init__(self, kind: str, keys: list[str], is_unique=False):
        """The keys can be nested, e.g., `user.address.street`."""
        self.kind = kind
        self.keys = keys
        self.is_unique = is_unique

#region Csv cache

CsvRow = list

CsvJoin = tuple[int, ...]
"""Represents indexes of rows that has to match."""

class CsvType(Enum):
    STRING = 'string'
    INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    DATETIME = 'datetime'
    DATE = 'date'
    JSON = 'json'
    STRING_LIST = 'string_list'

class CsvTable:
    def __init__(self, name: str, properties: list[CsvType]):
        self.name = name
        self.properties = properties
        """The order of properties should match the order of columns in the csv file."""

class MongoDocument:
    """Used for converting flat csv tables into nested mongodb documents."""
    def __init__(self, source: CsvTable, properties: dict[str, 'MongoProperty']):
        self.source = source
        self.properties = properties

class ComplexProperty:
    def __init__(self, document: MongoDocument, parent_join: CsvJoin, child_join: CsvJoin, is_array: bool):
        self.document = document
        self.parent_join = parent_join
        """An index from the parent's row that should match the child value."""
        self.child_join = child_join
        """An index from the child's row that should match the parent value."""
        self.is_array = is_array

MongoProperty = int | ComplexProperty
"""If this is a number, it represents the index of the column in the csv table."""

CsvIndexKey = tuple[str, ...]
CsvIndex = dict[CsvIndexKey, CsvRow | list[CsvRow]]
"""Map of key -> value / values (depending whether the index is unique or not)."""

class CachedCsvTable:

    """Stores flat data of the given csv table."""
    def __init__(self, csv_table: CsvTable):
        self.__csv_table = csv_table
        self.__rows = list[CsvRow]()
        self.__indexes = dict[CsvJoin, CsvIndex]()
        """Map of key column index -> index."""
        self.__indexes_uniqueness = dict[CsvJoin, bool]()
        """Whether is the given index unique."""

    def add_index(self, columns: CsvJoin, is_unique: bool):
        self.__indexes[columns] = {}
        self.__indexes_uniqueness[columns] = is_unique

    def load(self, import_directory: str):
        progress = ProgressTracker.unlimited()
        progress.start(f'Loading data for kind "{self.__csv_table.name}"... ')

        path = os.path.join(import_directory, self.__csv_table.name + '.tbl')

        with open_input(path) as file:
            reader = csv.reader(file, lineterminator='\n', delimiter='|')
            for row in reader:
                data = []
                for index, type in enumerate(self.__csv_table.properties):
                    data.append(csv_value_to_mongo(row[index], type))

                self.__rows.append(data)
                progress.track()

            progress.finish()

        for col_index in self.__indexes.keys():
            self.__create_index(col_index)

    def __create_index(self, columns: CsvJoin):
        values = self.__indexes[columns]
        is_unique = self.__indexes_uniqueness[columns]

        progress = ProgressTracker.limited(len(self.__rows))
        progress.start(f'   - Creating index on columns {columns}... ')

        for row in self.__rows:
            key = get_index_key_for_row(columns, row)
            if is_unique:
                if key in values:
                    # Just to be sure we have consistent data.
                    raise Exception(f'Duplicate key value "{key}" for unique index on "{self.__csv_table.name}"')
                values[key] = row
            else:
                values.setdefault(key, []).append(row)

            progress.track()

        progress.finish()

    def get_all(self) -> list[CsvRow]:
        return self.__rows

    def get_unique(self, columns: CsvJoin, key: CsvIndexKey) -> CsvRow:
        """Not very safe ... make sure you have defined the index uniqueness correctly ..."""
        return self.__indexes[columns][key]

    def get_non_unique(self, columns: CsvJoin, key: CsvIndexKey) -> list[CsvRow]:
        """Not very safe ... make sure you have defined the index uniqueness correctly ..."""
        return self.__indexes[columns].get(key, [])

def get_index_key_for_row(columns: CsvJoin, row: CsvRow) -> CsvIndexKey:
    return tuple(row[i] for i in columns)

TRUE_VALUES = { 'true', 't', '1', 'yes', 'y' }
FALSE_VALUES = { 'false', 'f', '0', 'no', 'n' }

def csv_value_to_mongo(value: str, type: CsvType):
    value = value.strip()

    if value == '':
        return None
    if type == CsvType.STRING:
        return value
    if type == CsvType.INT:
        return int(value)
    if type == CsvType.FLOAT:
        return float(value)
    if type == CsvType.BOOL:
        v = value.lower()
        if v in TRUE_VALUES:
            return True
        if v in FALSE_VALUES:
            return False
        raise ValueError(f'Invalid bool: {value}')
    if type == CsvType.DATETIME:
        return datetime.fromisoformat(value)
    if type == CsvType.DATE:
        # MongoDB doesn't have a separate date type, so we store dates as datetimes with time set to 00:00:00.
        return datetime.fromisoformat(value)
    if type == CsvType.JSON:
        return json.loads(value)
    if type == CsvType.STRING_LIST:
        return [x.strip() for x in value.split(',')]

postgres_to_csv_types: dict[str, CsvType] = {
    # Add more types as needed, but these should be enough for our current datasets.
    'BOOLEAN': CsvType.BOOL,

    'INTEGER': CsvType.INT,
    'SMALLINT': CsvType.INT,
    'BIGINT': CsvType.INT,

    'DECIMAL': CsvType.FLOAT,
    'FLOAT8': CsvType.FLOAT,

    'TEXT': CsvType.STRING,
    'CHAR': CsvType.STRING,
    'VARCHAR': CsvType.STRING,

    'DATE': CsvType.DATE,
    'TIMESTAMPTZ': CsvType.DATETIME,

    'JSONB': CsvType.JSON,
}

#endregion
#region Postgres builder

class MongoPostgresBuilder:

    @staticmethod
    def create(postgres: dict[str, list[PostgresColumn]]) -> 'MongoPostgresBuilder':
        builder = MongoPostgresBuilder()
        builder.__load_from_postgres(postgres)
        return builder

    def __load_from_postgres(self, postgres: dict[str, list[PostgresColumn]]):
        self.__csv_tables = {}
        self.__key_mapper = KeyToIndexMap()
        """A map of postgres property name -> csv column index for each kind. Just for convenience."""

        for kind, columns in postgres.items():
            self.__csv_tables[kind] = _postgres_to_csv_table(kind, columns)
            self.__key_mapper.add_kind(kind, columns)

    def document(self, csv_kind: str, properties: dict[str, 'MongoPropertyInit']) -> MongoDocument:
        """Convenience method to define documents from their csv sources."""
        table = self.__get_csv_table(csv_kind)

        mapped_properties = dict[str, MongoProperty]()
        for key, property in properties.items():
            mapped_properties[key] = self.__key_mapper.map_property(csv_kind, property)

        return MongoDocument(table, mapped_properties)

    def nested(self, csv_kind: str, properties: dict[str, 'MongoPropertyInit'], parent_join: int | str, child_join: int | str, is_array = False) -> 'ComplexPropertyInit':
        """Convenience method to define nested properties."""
        document = self.document(csv_kind, properties)
        return self.nest(document, parent_join, child_join, is_array)

    def nest(self, document: MongoDocument, parent_join: int | str, child_join: int | str, is_array = False) -> 'ComplexPropertyInit':
        """Convenience method to define nested properties."""
        return ComplexPropertyInit(document, parent_join, child_join, is_array)

    def plain_copy(self, csv_kind: str) -> MongoDocument:
        """Convenience method to define a document with all properties mapped directly from the csv (no nesting)."""
        table = self.__get_csv_table(csv_kind)
        # For some reason, dict[str, int] can't be assigned to dict[str, MongoProperty] even though int is a valid MongoProperty. However, we still want to copy the dict, so we just might as well ...
        properties: dict[str, MongoProperty] = {name: index for name, index in self.__key_mapper.get_all_properties(csv_kind).items()}

        return MongoDocument(table, properties)

    def __get_csv_table(self, csv_kind: str) -> CsvTable:
        table = self.__csv_tables.get(csv_kind)
        if not table:
            raise ValueError(f'CSV table not found for kind: {csv_kind}')
        return table

type_pattern = re.compile(r'([a-zA-Z0-9_()]+)')

def _postgres_to_csv_table(kind: str, columns: list[PostgresColumn]) -> CsvTable:
    properties = []

    for column in columns:
        match = type_pattern.match(column.type)
        if not match:
            raise ValueError(f'Invalid Postgres type: {column.type}')
        type_name = match.group(1)

        type = postgres_to_csv_types.get(type_name)
        if not type:
            raise ValueError(f'Unsupported Postgres type: {column.type}')

        properties.append(type)

    return CsvTable(kind, properties)

class KeyToIndexMap:
    """Enables using string keys (e.g., from postgres) instead of indexes."""
    def __init__(self):
        self.__kinds = dict[str, dict[str, int]]()

    def add_kind(self, kind: str, columns: list[PostgresColumn]):
        map = {}
        for index, column in enumerate(columns):
            map[column.name] = index
        self.__kinds[kind] = map

    def map_property(self, kind: str, init: 'MongoPropertyInit') -> MongoProperty:
        if isinstance(init, int) or isinstance(init, ComplexProperty):
            return init
        if isinstance(init, str):
            return self.map_simple(kind, init)
        return self.map_complex(kind, init)

    def map_simple(self, kind: str, column: str) -> int:
        column_index = self.__kinds[kind].get(column)
        if column_index is None:
            raise ValueError(f'Column with name "{column}" not found in CSV table for kind "{kind}".')

        return column_index

    def map_complex(self, kind: str, init: 'ComplexPropertyInit') -> ComplexProperty:
        parent_join = self.map_join(kind, init.parent_join)
        child_join = self.map_join(init.document.source.name, init.child_join)

        return ComplexProperty(init.document, parent_join, child_join, init.is_array)

    def map_join(self, kind: str, join: 'CsvJoinInit') -> CsvJoin:
        if isinstance(join, int):
            return (join,)
        if isinstance(join, str):
            return (self.map_simple(kind, join),)
        return tuple((column if isinstance(column, int) else self.map_simple(kind, column)) for column in join)

    def get_all_properties(self, kind: str) -> dict[str, int]:
        """Returns a map of { property_name: index }. Useful for copying all properties from a csv table."""
        return self.__kinds[kind]

CsvJoinInit = str | int | tuple[str | int, ...]
"""Either string | int (for one join) or a tuple of such (for multiple keys). Strings are interpreted as keys in the csv table, integers are indexes in such table."""

class ComplexPropertyInit:
    """A helper class to store the parameters for initializing a ComplexProperty."""
    def __init__(self, document: MongoDocument, parent_join: CsvJoinInit, child_join: CsvJoinInit, is_array: bool):
        self.document = document
        self.parent_join = parent_join
        self.child_join = child_join
        self.is_array = is_array

MongoPropertyInit = str | MongoProperty | ComplexPropertyInit

#endregion
