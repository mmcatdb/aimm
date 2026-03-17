import csv
from abc import ABC, abstractmethod
import datetime
from enum import Enum
import json
import os
from pymongo import ASCENDING
from common.utils import ProgressTracker, print_warning
from common.loaders.postgres_loader import ColumnSchema
from common.drivers import MongoDriver
import re

class MongoLoader(ABC):
    """A class to load data into a Mongo database."""
    def __init__(self, driver: MongoDriver):
        self._driver = driver

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the loader (for display purposes)."""
        pass

    @abstractmethod
    def _get_schemas(self) -> 'list[DocumentSchema]':
        """Returns the schemas for each entity kind."""
        pass

    @abstractmethod
    def _get_indexes(self) -> list['IndexSchema']:
        """Returns the list of indexes to be created."""
        pass

    def run(self, import_directory: str, do_reset: bool):
        title = f'--- {self.name()} Mongo Loader ---'
        print(title)
        print(f'Connecting to Mongo at: {self._driver.config.host}:{self._driver.config.port}')
        print('-' * len(title) + '\n')

        self._import_directory = import_directory
        self._build_cache()

        if do_reset:
            print('\nResetting database...')
            self.__reset_database()
            print('Database reset completed.')

        print('\nCreating constraints...')
        for index in self._get_indexes():
            self.__create_index(index)
        # TODO some other indexes?
        print('Constraints created.')

        print('\nLoading data...')
        for schema in self._get_schemas():
            self.__populate_kind(schema)
        print('Data loading completed.')

    def __reset_database(self):
        database = self._driver.database()
        collection_names = database.list_collection_names()
        for entity in reversed(collection_names):
            try:
                database.drop_collection(entity)
                print(f'Collection "{entity}" has been dropped in MongoDB.')
            except Exception as e:
                print_warning(f'Skipping delete for "{entity}"', e)

    def __create_index(self, index: 'IndexSchema'):
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

    def __populate_kind(self, schema: 'DocumentSchema'):
        collection_name = schema.source.kind
        collection = self._driver.collection(collection_name)
        rows = self._cache[schema.source.kind].get_all()

        progress = ProgressTracker.limited(len(rows))
        progress.start(f'Loading collection "{collection_name}"... ')

        for row in rows:
            document = self.__create_mongo_document(schema, row)
            collection.insert_one(document)
            progress.track()

        progress.finish()

    def __create_mongo_document(self, schema: 'DocumentSchema', row: 'CsvRow') -> dict:
        output = {}

        for key, property in schema.properties.items():
            if isinstance(property, int):
                output[key] = row[property]
                continue

            child = property
            table = self._cache[child.schema.source.kind]

            parent_index_key = get_index_key_for_row(child.parent_join, row)

            if child.is_array:
                child_rows = table.get_non_unique(child.child_join, parent_index_key)
                child_value = [self.__create_mongo_document(child.schema, child_row) for child_row in child_rows]
            else:
                child_row = table.get_unique(child.child_join, parent_index_key)
                child_value = self.__create_mongo_document(child.schema, child_row)

            output[key] = child_value

        return output

    def _build_cache(self):
        self._cache: dict[str, CachedCsvTable] = {}

        for schema in self._get_schemas():
            self._find_required_tables(schema, None, None)

        # Verify that all files exist in the import directory.
        print(f'Using .tbl files directly from the import directory: "{self._import_directory}"\n')

        for kind in self._cache.values():
            path = os.path.join(self._import_directory, kind.schema.kind + '.tbl')
            if not os.path.isfile(path):
                raise Exception(f'Required file not found in import directory: {path}')

        # Load all kinds. The order doesn't matter, since they will reference each other via the cache, not via the database.
        for kind in self._cache.values():
            kind.load(self._import_directory)

    def _find_required_tables(self, schema: 'DocumentSchema', index_by: 'CsvJoin | None', is_array: bool | None):
        table = self._cache.get(schema.source.kind)
        if not table:
            table = CachedCsvTable(schema.source)
            self._cache[schema.source.kind] = table

        if index_by is not None:
            table.add_index(index_by, not is_array)

        for property in schema.properties.values():
            if isinstance(property, ComplexProperty):
                self._find_required_tables(property.schema, property.child_join, property.is_array)

class IndexSchema:
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

class CsvSchema:
    def __init__(self, kind: str, properties: list[CsvType]):
        self.kind = kind
        self.properties = properties
        """The order of properties should match the order of columns in the csv file."""

class DocumentSchema:
    """Used for converting flat csv tables into nested mongodb documents."""
    def __init__(self, source: CsvSchema, properties: dict[str, 'MongoProperty']):
        self.source = source
        self.properties = properties

class ComplexProperty:
    def __init__(self, schema: DocumentSchema, parent_join: CsvJoin, child_join: CsvJoin, is_array: bool):
        self.schema = schema
        self.parent_join = parent_join
        """An index from the parent's row that should match the child value."""
        self.child_join = child_join
        """An index from the child's row that should match the parent value."""
        self.is_array = is_array

MongoProperty = int | ComplexProperty
"""If this is a number, it represents the index of the column in the csv schema."""

CsvIndexKey = tuple[str, ...]
CsvIndex = dict[CsvIndexKey, CsvRow | list[CsvRow]]
"""Map of key -> value / values (depending whether the index is unique or not)."""

class CachedCsvTable:
    """Stores flat data of the given csv table."""
    def __init__(self, schema: CsvSchema):
        self.schema = schema
        self.__rows: list[CsvRow] = []
        self.__indexes: dict[CsvJoin, CsvIndex] = {}
        """Map of key column index -> index."""
        self.__indexes_uniqueness: dict[CsvJoin, bool] = {}
        """Whether is the given index unique."""

    def add_index(self, columns: CsvJoin, is_unique: bool):
        self.__indexes[columns] = {}
        self.__indexes_uniqueness[columns] = is_unique

    def load(self, import_directory: str):
        progress = ProgressTracker.unlimited()
        progress.start(f'Loading data for kind "{self.schema.kind}"... ')

        path = os.path.join(import_directory, self.schema.kind + '.tbl')

        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter='|')

            for row in reader:
                data = []
                for index, type in enumerate(self.schema.properties):
                    data.append(csv_value_to_mongo(row[index], type))

                self.__rows.append(data)
                progress.track()

            progress.finish()

        for col_index in self.__indexes.keys():
            self._create_index(col_index)

    def _create_index(self, columns: CsvJoin):
        values = self.__indexes[columns]
        is_unique = self.__indexes_uniqueness[columns]

        progress = ProgressTracker.limited(len(self.__rows))
        progress.start(f'   - Creating index on columns {columns}... ')

        for row in self.__rows:
            key = get_index_key_for_row(columns, row)
            if is_unique:
                if key in values:
                    # Just to be sure we have consistent data.
                    raise Exception(f'Duplicate key value "{key}" for unique index on "{self.schema.kind}"')
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
        return self.__indexes[columns][key]

def get_index_key_for_row(columns: CsvJoin, row: CsvRow) -> CsvIndexKey:
    return tuple(row[i] for i in columns)

TRUE_VALUES = { 'true', 't', '1', 'yes', 'y' }
FALSE_VALUES = { 'false', 'f', '0', 'no', 'n' }

def csv_value_to_mongo(value: str, typ: CsvType):
    value = value.strip()

    if value == '':
        return None
    if typ == CsvType.STRING:
        return value
    if typ == CsvType.INT:
        return int(value)
    if typ == CsvType.FLOAT:
        return float(value)
    if typ == CsvType.BOOL:
        v = value.lower()
        if v in TRUE_VALUES:
            return True
        if v in FALSE_VALUES:
            return False
        raise ValueError(f'Invalid bool: {value}')
    if typ == CsvType.DATETIME:
        return datetime.datetime.fromisoformat(value)
    if typ == CsvType.DATE:
        # MongoDB doesn't have a separate date type, so we store dates as datetimes with time set to 00:00:00.
        return datetime.datetime.fromisoformat(value)
    if typ == CsvType.JSON:
        return json.loads(value)
    if typ == CsvType.STRING_LIST:
        return [x.strip() for x in value.split(',')]

postgres_to_csv_types: dict[str, CsvType] = {
    # Add more types as needed, but these should be enough for our current datasets.
    'BOOLEAN': CsvType.BOOL,
    'CHAR': CsvType.STRING,
    'DATE': CsvType.DATE,
    'DECIMAL': CsvType.FLOAT,
    'INTEGER': CsvType.INT,
    'JSONB': CsvType.JSON,
    'SMALLINT': CsvType.INT,
    'TEXT': CsvType.STRING,
    'TIMESTAMPTZ': CsvType.DATETIME,
    'VARCHAR': CsvType.STRING,
}

#endregion
#region Postgres builder

class MongoPostgresBuilder:
    @staticmethod
    def create(postgres: dict[str, list[ColumnSchema]]) -> 'MongoPostgresBuilder':
        builder = MongoPostgresBuilder()
        builder.__load_from_postgres(postgres)
        return builder

    def __load_from_postgres(self, postgres: dict[str, list[ColumnSchema]]):
        self.__csv_schemas = {}
        self.__key_mapper = KeyToIndexMap()
        """A map of postgres property name -> csv column index for each kind. Just for convenience."""

        for kind, columns in postgres.items():
            self.__csv_schemas[kind] = _postgres_to_csv_schema(kind, columns)
            self.__key_mapper.add_kind(kind, columns)

    def document(self, csv_kind: str, properties: dict[str, 'MongoPropertyInit']) -> DocumentSchema:
        """Convenience method to define document schemas from their csv source."""
        csv_schema = self.__get_csv_schema(csv_kind)

        mapped_properties: dict[str, MongoProperty] = {}
        for key, property in properties.items():
            mapped_properties[key] = self.__key_mapper.map_property(csv_kind, property)

        return DocumentSchema(csv_schema, mapped_properties)

    def nested(self, csv_kind: str, properties: dict[str, 'MongoPropertyInit'], parent_join: int | str, child_join: int | str, is_array = False) -> 'ComplexPropertyInit':
        """Convenience method to define nested properties."""
        schema = self.document(csv_kind, properties)
        return self.nest(schema, parent_join, child_join, is_array)

    def nest(self, schema: DocumentSchema, parent_join: int | str, child_join: int | str, is_array = False) -> 'ComplexPropertyInit':
        """Convenience method to define nested properties."""
        return ComplexPropertyInit(schema, parent_join, child_join, is_array)

    def plain_copy(self, csv_kind: str) -> DocumentSchema:
        """Convenience method to define a document schema with all properties mapped directly from the csv (no nesting)."""
        csv_schema = self.__get_csv_schema(csv_kind)
        # For some reason, dict[str, int] can't be assigned to dict[str, MongoProperty] even though int is a valid MongoProperty. However, we still want to copy the dict, so we just might as well ...
        properties: dict[str, MongoProperty] = {name: index for name, index in self.__key_mapper.get_all_properties(csv_kind).items()}

        return DocumentSchema(csv_schema, properties)

    def __get_csv_schema(self, csv_kind: str) -> CsvSchema:
        csv_schema = self.__csv_schemas.get(csv_kind)
        if not csv_schema:
            raise ValueError(f'CSV schema not found for kind: {csv_kind}')
        return csv_schema

type_pattern = re.compile(r'([A-Z]+)')

def _postgres_to_csv_schema(kind: str, columns: list[ColumnSchema]) -> CsvSchema:
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

    return CsvSchema(kind, properties)

class KeyToIndexMap:
    """Enables using string keys (e.g., from postgres) instead of indexes."""
    def __init__(self):
        self.__kinds: dict[str, dict[str, int]] = {}

    def add_kind(self, kind: str, columns: list[ColumnSchema]):
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
            raise ValueError(f'Column with name "{column}" not found in CSV schema for kind "{kind}".')

        return column_index

    def map_complex(self, kind: str, init: 'ComplexPropertyInit') -> ComplexProperty:
        parent_join = self.map_join(kind, init.parent_join)
        child_join = self.map_join(init.schema.source.kind, init.child_join)

        return ComplexProperty(init.schema, parent_join, child_join, init.is_array)

    def map_join(self, kind: str, join: 'CsvJoinInit') -> CsvJoin:
        if isinstance(join, int):
            return (join,)
        if isinstance(join, str):
            return (self.map_simple(kind, join),)
        return tuple((column if isinstance(column, int) else self.map_simple(kind, column)) for column in join)

    def get_all_properties(self, kind: str) -> dict[str, int]:
        """Returns a map of { property_name: index }. Useful for copying all properties from a csv schema."""
        return self.__kinds[kind]

CsvJoinInit = str | int | tuple[str | int, ...]
"""Either string | int (for one join) or a tuple of such (for multiple keys). Strings are interpreted as keys in the csv schema, integers are indexes in such schema."""

class ComplexPropertyInit:
    """A helper class to store the parameters for initializing a ComplexProperty."""
    def __init__(self, schema: DocumentSchema, parent_join: CsvJoinInit, child_join: CsvJoinInit, is_array: bool):
        self.schema = schema
        self.parent_join = parent_join
        self.child_join = child_join
        self.is_array = is_array

MongoPropertyInit = str | MongoProperty | ComplexPropertyInit

#endregion
