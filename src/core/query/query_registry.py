from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
import random
from typing import Any, Generic, TypeVar, cast
from core.config import GLOBAL_RNG_SEED
from core.drivers import DriverType
from .query_id import SchemaName, TemplateName
from .query_instance import TQuery, QueryInstance
from .query_template import QueryGenerator, QueryTemplate

class QueryMetadata():
    def __init__(self, name: str, weight: float, title: str | None):
        self.name = name
        self.weight = weight
        self.title = title

QueryFunction = TypeVar('QueryFunction', bound=Callable[..., Any])

def query(name: str, title: str | None = None) -> Callable[[QueryFunction], QueryFunction]:
    """Decorator to mark a method as a query generator. The method should take a `QueryRegistry` as input and return a `TQuery`."""
    return weighted_query(name, 1.0, title)

def weighted_query(name: str, weight: float, title: str | None = None) -> Callable[[QueryFunction], QueryFunction]:
    """Decorator to mark a method as a query generator. The method should take a `QueryRegistry` as input and return a `TQuery`."""
    def decorator(function: QueryFunction) -> QueryFunction:
        setattr(function, '_is_query', True)
        setattr(function, '_query_metadata', QueryMetadata(name, weight, title))
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

    def generate_queries(self, scale: float, num_queries: int) -> list[QueryInstance[TQuery]]:
        """Generates `num_queries` query instances. At least one query will be generated for each template."""
        queries = list[QueryInstance[TQuery]]()
        templates = list(self._get_templates().values())
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
            self._query(metadata.name, metadata.weight, metadata.title, function)

        self._register_queries()

        output = self.__collected_templates

        self.__collected_templates = None

        return output

    def _register_queries(self):
        """Override this to register queries programmatically using the `_query` method."""
        pass

    def _query(self, name: str, weight: float, title: str | None, function: Callable[[QueryRegistry[TQuery]], TQuery]):
        """Register query programatically.

        Works the same way as the `@query` decorator but can be used dynamically.
        Has to be called within `_register_queries`.
        """
        if self.__collected_templates is None:
            raise Exception('_query can only be called within _register_queries method.')

        generator = self.__create_query_generator(function)

        self.__collected_templates.append(QueryTemplate(
            driver=self.driver,
            schema=self.schema,
            name=name,
            weight=weight,
            title=title,
            generator=generator,
        ))

    def __create_query_generator(self, function: 'Callable[[QueryRegistry[TQuery]], TQuery]') -> QueryGenerator[TQuery]:
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

    def _param_int(self, name: str, min_value: int, max_value: int):
        return self._param(name, lambda: self._rng_int(min_value, max_value))

    def _param_float(self, name: str, min_value: float, max_value: float):
        return self._param(name, lambda: self._rng.uniform(min_value, max_value))

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
