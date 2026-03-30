from collections.abc import Callable
from math import ceil
import random
from typing import Any, Generic, TypeVar, cast
from common.utils import print_warning
from common.config import GLOBAL_RNG_SEED

TQuery = TypeVar('TQuery')

class QueryDef(Generic[TQuery]):
    def __init__(self, category: str, index: int, weight: float, title: str | None, generator: Callable[[bool], TQuery]):
        self.weight = weight
        self.category = category
        """A short string grouping queries by expected operators (e.g., 'join', 'agg', ...)."""
        self.index = index
        """Index of the query within its category."""
        self.id = f'{category}:{index}'
        self.title = title
        self._generator = generator
        """If true, produces a template with parameter placeholders instead of actual values. If false, produces a fully instantiated query."""

    @staticmethod
    def create_from_content(category: str, index: int, weight: float, title: str | None, content: TQuery) -> 'QueryDef[TQuery]':
        """Creates a QueryDef with a simple generator that always returns the same content."""
        return QueryDef(category, index, weight, title, lambda _: content)

    def label(self) -> str:
        """Returns a human-readable label for this query, combining ID and title."""
        title = self.title if self.title else '(no title)'
        return f'{self.id} - {title}'

    def generate(self) -> TQuery:
        """Generates a query by filling the template with parameters."""
        return self._generator(False)

    def template(self) -> TQuery:
        """Returns a representation of the query template with placeholders for parameters. Useful for debugging."""
        return self._generator(True)

class QueryMetadata():
    def __init__(self, category: str, weight: float, title: str | None):
        self.category = category
        self.weight = weight
        self.title = title

QueryFunction = TypeVar('QueryFunction', bound=Callable[..., Any])

def query(category: str, weight: float, title: str | None = None):
    """Decorator to mark a method as a query generator. The method should take a `QueryRegistry` as input and return a `TQuery`."""
    def decorator(function: QueryFunction) -> QueryFunction:
        setattr(function, '_is_query', True)
        setattr(function, '_query_metadata', QueryMetadata(category, weight, title))
        return function

    return decorator

TParam = TypeVar('TParam')

QueryDefMap = dict[int, QueryDef[TQuery]]
"""Map from `id()` of a generated query to its definition. Useful for debugging and analysis."""

class QueryRegistry(Generic[TQuery]):
    def __init__(self):
        self.__is_template = False
        self.__query_defs: dict[str, QueryDef[TQuery]] | None = None
        self._rng = random.Random(GLOBAL_RNG_SEED)

    def get_query_defs(self) -> list[QueryDef[TQuery]]:
        """Returns all collected query definitions. Useful for final evaluation or debugging."""
        return list(self._query_defs().values())

    def get_query_def(self, id: str) -> QueryDef[TQuery] | None:
        """Returns a single query by its ID (or None if not found). Useful for debugging specific queries."""
        return self._query_defs().get(id)

    #region Generation

    def generate_test_queries(self) -> tuple[QueryDefMap[TQuery], list[TQuery]]:
        """Generates one query for each definition in the original order. Useful for testing and debugging."""
        def_map = QueryDefMap[TQuery]()
        queries = list[TQuery]()
        for def_ in self._query_defs().values():
            query = def_.generate()
            queries.append(query)
            def_map[id(query)] = def_

        return def_map, queries

    def generate_queries(self, num_queries: int, train_split: float) -> tuple[QueryDefMap[TQuery], list[TQuery], list[TQuery]]:
        """
        Generates at least `num_queries` queries with parameter variations. The actual number may be higher due to query weighting.
        At least one query will be generated for each definition.

        The queries are split into train and validation sets accordingly. The split happens at the definition level, so all queries generated from a particular definition will be in the same set.
        Each category is split independently to ensure all categories are represented in both sets.

        For this reason, query weights should not vary too much within the same category. And each category should have at least two definitions.
        """
        defs = self._query_defs().values()

        groups = {}
        for def_ in defs:
            if def_.category not in groups:
                groups[def_.category] = []
            groups[def_.category].append(def_)

        total_weight = sum(def_.weight for def_ in defs)
        if total_weight == 0:
            print_warning('Total query weight is zero. All queries will be generated with the same frequency regardless of their specified weights.')
            total_weight = len(defs)

        queries_per_weight = num_queries / total_weight
        map_output = QueryDefMap[TQuery]()
        train_output = list[TQuery]()
        val_output = list[TQuery]()

        for group in groups.values():
            def_map, train, val = self.__generate_for_group(group, queries_per_weight, train_split)
            map_output.update(def_map)
            train_output.extend(train)
            val_output.extend(val)

        self._rng.shuffle(train_output)
        self._rng.shuffle(val_output)

        return map_output, train_output, val_output

    def __generate_for_group(self, group: list[QueryDef[TQuery]], queries_per_weight: float, train_split: float) -> tuple[QueryDefMap[TQuery], list[TQuery], list[TQuery]]:
        group_weight = sum(def_.weight for def_ in group)

        if len(group) < 2 or group_weight == 0:
            print_warning(f'Category "{group[0].category}" has only {len(group)} definition(s) or zero weight. Cannot split into train/validation sets as intended. All queries from this category will be in the train set.')
            def_map, queries = self.__generate_list(group, queries_per_weight)
            return def_map, queries, []

        shuffled = group[:]
        self._rng.shuffle(shuffled)

        train_weight = 0
        train_weight_target = group_weight * train_split
        index = 0
        for def_ in shuffled:
            index += 1
            train_weight += def_.weight
            if train_weight >= train_weight_target:
                break

        if (index == len(shuffled)):
            # Ensure at least one definition goes to the validation set
            index -= 1
            train_weight -= shuffled[index].weight

        val_weight = group_weight - train_weight

        train_defs = shuffled[:index]
        val_defs = shuffled[index:]

        # Normalize train/validation weights to match the exact train_split as closely as possible
        num_queries = queries_per_weight * group_weight
        train_qpw = train_split * num_queries / train_weight
        val_qpw = (1 - train_split) * num_queries / val_weight

        train_map, train_queries = self.__generate_list(train_defs, train_qpw)
        val_map, val_queries = self.__generate_list(val_defs, val_qpw)

        train_map.update(val_map)

        return train_map, train_queries, val_queries

    def __generate_list(self, defs: list[QueryDef[TQuery]], queries_per_weight: float) -> tuple[QueryDefMap[TQuery], list[TQuery]]:
        output = list[TQuery]()
        def_map = QueryDefMap[TQuery]()
        for def_ in defs:
            for _ in range(max(ceil(queries_per_weight * def_.weight), 1)):
                query = def_.generate()
                output.append(query)
                def_map[id(query)] = def_

        return def_map, output

    #endregion
    #region Collection

    def _param(self, name: str, generator: Callable[[], TParam]) -> TParam | str:
        """Base method for defining a parameter. If in template mode, returns a placeholder string instead of generating an actual value."""
        return '{{' + name + '}}' if self.__is_template else generator()

    def _query_defs(self) -> dict[str, QueryDef[TQuery]]:
        """Returns a list of query definitions collected from the methods of this class."""
        if self.__query_defs is None:
            self.__query_defs = {q.id: q for q in self.__collect_queries()}

        return self.__query_defs

    def __create_query_generator(self, function: 'Callable[[QueryRegistry[TQuery]], TQuery]') -> Callable[[bool], TQuery]:
        def generator(template_mode: bool) -> TQuery:
            self.__is_template = template_mode
            return function(self)

        return generator

    @classmethod
    def __get_attributes(cls):
        return cls.__dict__

    def __collect_queries(self) -> list[QueryDef[TQuery]]:
        queries = list[QueryDef[TQuery]]()
        category_indexes = dict[str, int]()

        for index, attribute in enumerate(self.__get_attributes().values()):
            if not callable(attribute) or not getattr(attribute, '_is_query', False):
                continue

            function = cast('Callable[[QueryRegistry[TQuery]], TQuery]', attribute)
            metadata = cast(QueryMetadata, getattr(function, '_query_metadata'))
            generator = self.__create_query_generator(function)

            category = metadata.category
            index = category_indexes.get(category, 0)
            category_indexes[category] = index + 1

            queries.append(QueryDef(
                category=category,
                index=index,
                weight=metadata.weight,
                title=metadata.title,
                generator=generator,
            ))

        return queries

    #endregion
