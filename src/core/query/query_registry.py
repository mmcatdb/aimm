from __future__ import annotations
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
import random
from typing import Any, Generic, TypeVar, cast
from typing_extensions import Self
from core.config import GLOBAL_RNG_SEED
from core.drivers import DriverType
from .query_id import SchemaName, TemplateName
from .query_instance import TQuery, QueryInstance
from .query_template import QueryGenerator, QueryTemplate

class QueryMetadata():
    def __init__(self, name: str, title: str, weight: float, is_write: bool):
        self.name = name
        self.title = title
        self.weight = weight
        self.is_write = is_write

QueryFunction = TypeVar('QueryFunction', bound=Callable[..., Any])

def query(name: str, title: str, weight: float = 1.0) -> Callable[[QueryFunction], QueryFunction]:
    """Decorator to mark a method as a query generator. The method should take a `QueryRegistry` as input and return a `TQuery`."""
    return _common_query(name, title, weight, is_write=False)

def write_query(name: str, title: str, weight: float = 1.0) -> Callable[[QueryFunction], QueryFunction]:
    """Decorator to mark a method as a write query generator. The method should take a `QueryRegistry` as input and return a `TQuery`. Write queries are meant to modify the database."""
    return _common_query(name, title, weight, is_write=True)

def _common_query(name: str, title: str, weight: float, is_write: bool) -> Callable[[QueryFunction], QueryFunction]:
    def decorator(function: QueryFunction) -> QueryFunction:
        setattr(function, '_is_query', True)
        setattr(function, '_query_metadata', QueryMetadata(name, title, weight, is_write))
        return function

    return decorator

class ValueType(Enum):
    STRING = 'string'
    NUMBER = 'number'
    DATE = 'date'

TParam = TypeVar('TParam')
TListItem = TypeVar('TListItem')

class QueryRegistry(Generic[TQuery]):

    def __init__(self, driver: DriverType, schema: SchemaName):
        self.driver = driver
        self.schema = schema
        self._rng = random.Random(GLOBAL_RNG_SEED)
        self._scale: float
        """The current scale factor for which the parameters are being generated. Set when generating queries."""
        self.__is_raw = False
        # It's better to map the templates by name instead of their full ID because part of the ID is already contained in the registry (driver and schema).
        # We would probably need to parse the template name from the ID anyway for the generated queries.
        self.__templates: dict[TemplateName, QueryTemplate[TQuery]] | None = None

    def __set_scale(self, scale: float):
        """If needed, sets the current scale factor for which the parameters are being generated. Is used lazily by the parameter generators."""
        if not hasattr(self, '_scale') or self._scale != scale:
            self._scale = scale
            self._setup_cache()
            self._used_ints = dict[str, set[int]]()
            """Used for generating unique integers. Maps a key (e.g., parameter name) to the set of already used integers for that key."""

    def _setup_cache(self):
        """Called whenever the scale changes. Can be used to reset any cached data that depends on the scale."""

        # The logic behind this (and the previous) method is that some parameters might be expensive to generate.
        # E.g., we might need to query the database to find the available range of values for a certain parameter, which depends on the scale.
        # However, we still want to have the templates independent from the scale.
        # This way, we can generate the parameters lazily when they are needed, but still cache any expensive computations that depend on the scale.

        pass

    def get_template(self, name: TemplateName) -> QueryTemplate[TQuery] | None:
        """Returns a single query by its name (or None if not found). Useful for debugging specific queries."""
        # TODO Override this for database with dynamic templates.
        return self._get_templates().get(name)

    #region Generation

    def generate_queries(self, scale: float, num_queries: int, allow_write: bool) -> list[QueryInstance[TQuery]]:
        """Generates `num_queries` query instances. At least one query will be generated for each template."""
        queries = list[QueryInstance[TQuery]]()
        templates = list(self._get_templates().values())
        if not allow_write:
            templates = [t for t in templates if not t.is_write]

        if num_queries <= 0:
            num_queries = len(templates)

        for i in range(num_queries):
            template = templates[i % len(templates)]
            query = template.generate(scale, i // len(templates))
            queries.append(query)

        return queries

    def _get_templates(self) -> dict[TemplateName, QueryTemplate[TQuery]]:
        """Returns a list of query definitions collected from the methods of this class."""
        if self.__templates is None:
            self.__templates = {q.name: q for q in self.__collect_queries()}

        return self.__templates

    @classmethod
    def __get_attributes(cls):
        return cls.__dict__

    def __collect_queries(self) -> list[QueryTemplate[TQuery]]:
        self.__collected_templates = list[QueryTemplate[TQuery]]()

        for attribute in self.__get_attributes().values():
            if not callable(attribute) or not getattr(attribute, '_is_query', False):
                continue

            function = cast('Callable[[QueryRegistry[TQuery]], TQuery]', attribute)
            metadata = cast(QueryMetadata, getattr(function, '_query_metadata'))
            self.__register_query(metadata.name, metadata.title, metadata.weight, function, metadata.is_write)

        self._register_queries()

        output = self.__collected_templates

        self.__collected_templates = None

        return output

    def _register_queries(self):
        """Override this to register queries programmatically using the `_query` method."""
        pass

    def _register_write_queries(self):
        """Override this to register write queries programmatically using the `_query` method."""
        pass

    def _query(self, name: str, title: str, function: Callable[[Self], TQuery], weight: float = 1.0):
        """Register query programatically.

        Works the same way as the `@query` decorator but can be used dynamically.
        Has to be called within `_register_queries`.
        """
        self.__register_query(name, title, weight, function, is_write=False)

    def _write_query(self, name: str, title: str, function: Callable[[Self], TQuery], weight: float = 1.0):
        """Register write query programatically. Write queries are meant to modify the database.

        Works the same way as the `@query` decorator but can be used dynamically.
        Has to be called within `_register_queries`.
        """
        self.__register_query(name, title, weight, function, is_write=True)

    def __register_query(self, name: str, title: str, weight: float, function: Callable[[Self], TQuery], is_write: bool):
        if self.__collected_templates is None:
            raise Exception('Query can only be registered within _register_queries method.')

        generator = self.__create_query_generator(function)

        self.__collected_templates.append(QueryTemplate(
            driver=self.driver,
            schema=self.schema,
            name=name,
            weight=weight,
            title=title,
            is_write=is_write,
            generator=generator,
        ))

    def __create_query_generator(self, function: Callable[[Self], TQuery]) -> QueryGenerator[TQuery]:
        def generator(scale: float | None, is_raw: bool) -> TQuery:
            if scale is not None:
                self.__set_scale(scale)
            elif not is_raw:
                raise Exception('Scale must be provided when generating actual queries.')

            self.__is_raw = is_raw
            return function(self)

        return generator

    #endregion
    #region Params

    def _param(self, name: str, generator: Callable[[], TParam]) -> TParam | str:
        """Base method for defining a parameter. If in raw mode, returns a placeholder string instead of generating an actual value."""
        return '{{' + name + '}}' if self.__is_raw else generator()

    def _convert_scalar(self, value: Any, type: ValueType) -> Any:
        """Converts a scalar value to a representation suitable for queries."""
        if type == ValueType.STRING:
            return self._convert_string(value)
        elif type == ValueType.NUMBER:
            return str(value)
        elif type == ValueType.DATE:
            return self._convert_date(value)

    def _convert_string(self, value: str) -> Any:
        if self.driver == DriverType.POSTGRES or self.driver == DriverType.NEO4J:
            return f"'{value}'"

        # MongoDB doesn't need any conversion.
        return value

    def _convert_date(self, date: datetime) -> Any:
        """Converts a date to a representation suitable for queries."""
        if self.driver == DriverType.POSTGRES or self.driver == DriverType.NEO4J:
            return date.strftime('%Y-%m-%d')

        # MongoDB doesn't need any conversion.
        return date

    def _convert_array(self, array: list[Any], type: ValueType) -> Any:
        """Converts an array of values to a representation suitable for queries."""
        if self.driver == DriverType.POSTGRES or self.driver == DriverType.NEO4J:
            # The queries are supposed to put the correct brackets around.
            return ', '.join(map(lambda v: self._convert_scalar(v, type), array))

        # MongoDB doesn't need any conversion.
        return array

    def _rng_date(self, start_year=1992, end_year=1998) -> datetime:
        years = end_year - start_year + 1
        # Not the most accurate way to handle leap years, but good enough for random generation.
        seconds = years * 365 * 24 * 60 * 60
        return datetime(start_year, 1, 1) + timedelta(seconds = self._rng.randint(0, seconds))

    # Common utility methods for generating random parameters. Could be overridden by specific databases if needed.

    def _param_month(self):
        return self._param_int('month', 1, 12)

    def _param_date_minus_days(self, min_days: int, max_days: int):
        days = self._rng_int(min_days, max_days)
        return self._param('date', lambda: self._convert_date(datetime.now() - timedelta(days=days)))

    def _rng_int(self, min_value: int, max_value: int):
        return self._rng.randint(min_value, max_value)

    def _rng_unique_int(self, key: str, min_value: int, max_value: int):
        """Generates a random integer in the given range that is guaranteed to not be generated again for the same key (until the cache is cleared)."""
        used = self._used_ints.setdefault(key, set())
        if len(used) >= (max_value - min_value + 1):
            # There might be false positives but it's better than an infinite loop.
            raise Exception(f'No more unique integers available for key {key} in the range [{min_value}, {max_value}].')

        candidate = self._rng_int(min_value, max_value)
        while candidate in used:
            candidate = self._rng_int(min_value, max_value)
        used.add(candidate)
        return candidate

    def _param_int(self, name: str, min_value: int, max_value: int):
        return self._param(name, lambda: self._rng_int(min_value, max_value))

    def _param_float(self, name: str, min_value: float, max_value: float, round_digits: int | None = None):
        if round_digits is not None:
            generator = lambda: round(self._rng.uniform(min_value, max_value), round_digits)
        else:
            generator = lambda: self._rng.uniform(min_value, max_value)

        return self._param(name, generator)

    def _rng_int_array(self, min_value: int, max_value: int, min_count: int, max_count: int, is_unique: bool = False) -> list[int]:
        """min_count is NOT quarranteed if is_unique is True (even if there are enough unique values)."""
        count = self._rng_int(min_count, max_count)
        output = [self._rng_int(min_value, max_value) for _ in range(count)]
        if is_unique:
            # If we removed some duplicates, we might end up with less than min_count items. In that case, we just return what we have.
            output = list(set(output))

        return output

    def _param_choice(self, name: str, choices: list[TListItem]):
        return self._param(name, lambda: self._rng.choice(choices))

    def _param_limit(self, min_value: int = 10, max_order: int = 5):
        """Returns a random choice from [min_value * 2^n] for n in [0, ..., max_order]."""
        choices: list[int] = [min_value * (2 ** n) for n in range(max_order + 1)]
        return self._param_choice('limit', choices)

    def _param_skip(self, min_value: int = 10, max_value: int = 1000):
        return self._param_int('skip', min_value, max_value)

    def _param_int_array(self, name: str, max_value: int, min_count: int, max_count: int | None):
        """Useful for ids with the IN operator. If max_count is None, it will be set to min_count (i.e. fixed length arrays)."""
        if max_count is None:
            max_count = min_count

        return self._param(name, lambda: self._convert_array(
            self._rng_int_array(min_value=1, max_value=max_value, min_count=min_count, max_count=max_count, is_unique=True),
            ValueType.NUMBER,
        ))

    #endregion
