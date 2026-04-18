from __future__ import annotations
from math import ceil
import random
from typing import Generic, Protocol
from core.drivers import DriverType
from core.utils import print_warning
from .query_id import SchemaName, create_database_id, create_query_instance_id, TemplateName, TemplateId, create_template_id
from .query_instance import QueryInstance, TQuery, TQuery_cov

class QueryGenerator(Protocol[TQuery_cov]):
    def __call__(self, scale: float | None, is_raw: bool) -> TQuery_cov: ...
    """Creates a query content with parameters.

    If `is_raw` is true, uses parameter placeholders instead of actual values. If false, produces a fully instantiated query.
    The `scale` parameter should be provided when `is_raw` is false, so that the parameters can be generated according to the schema data scale. It can be ignored when `is_raw` is true, since the output will just contain placeholders.
    """

class QueryTemplate(Generic[TQuery]):
    """A template that can be instantiated with parameters to produce a `QueryInstance`."""

    def __init__(self, driver: DriverType, schema: SchemaName, name: TemplateName, weight: float, title: str | None, generator: QueryGenerator[TQuery]):
        self.driver = driver
        self.schema = schema
        self.name = name
        self.weight = weight
        """Used both for determining the frequency of query generation and the weight of the query during evaluation / MCTS."""
        self.id: TemplateId = create_template_id(driver, schema, name)
        self._title = title
        self._generator = generator

    @staticmethod
    def create_from_content(driver: DriverType, schema: SchemaName, name: TemplateName, weight: float, title: str | None, content: TQuery) -> QueryTemplate[TQuery]:
        """Creates a QueryTemplate with a simple generator that always returns the same content."""
        return QueryTemplate(driver, schema, name, weight, title, lambda scale, is_raw: content)

    def label(self) -> str:
        """Returns a human-readable label for this query, combining ID and title."""
        title = self._title if self._title else '(no title)'
        return f'{self.id} - {title}'

    def generate(self, scale: float, index: int) -> QueryInstance[TQuery]:
        """Generates a query by filling the template with parameters."""
        content = self._generator(scale, is_raw=False)
        database_id = create_database_id(self.driver, self.schema, scale)
        query_id = create_query_instance_id(database_id, self.name, index)
        return QueryInstance(query_id, self.label(), content)

    def raw(self) -> TQuery:
        """Returns a representation of the query template with placeholders for parameters. Useful for debugging."""
        return self._generator(None, is_raw=True)

#region Not used
# NOTE This part is not used right now, but we might want to restore it later.

class CategorizedQueryTemplate(QueryTemplate[TQuery]):
    """A QueryTemplate with an associated category and weight for generation frequency."""

    def __init__(self, driver: DriverType, schema: SchemaName, name: TemplateName, weight: float, title: str | None, generator: QueryGenerator[TQuery], category: str):
        super().__init__(driver, schema, name, weight, title, generator)
        self.category = category

class CategorizedQueryGenerator:

    def __init__(self, rng: random.Random, scale: float, train_split: float):
        self._rng = rng
        self._scale = scale
        self._train_split = train_split

    def generate_queries(self, templates: list[CategorizedQueryTemplate[TQuery]], num_queries: int) -> tuple[list[QueryInstance[TQuery]], list[QueryInstance[TQuery]]]:
        """
        Generates at least `num_queries` queries with parameter variations. The actual number may be higher due to query weighting.
        At least one query will be generated for each definition.

        The queries are split into train and validation sets accordingly. The split happens at the definition level, so all queries generated from a particular definition will be in the same set.
        Each category is split independently to ensure all categories are represented in both sets.

        For this reason, query weights should not vary too much within the same category. And each category should have at least two definitions.
        """
        groups = {}
        for template in templates:
            if template.category not in groups:
                groups[template.category] = []
            groups[template.category].append(template)

        total_weight = sum(template.weight for template in templates)
        if total_weight == 0:
            print_warning('Total query weight is zero. All queries will be generated with the same frequency regardless of their specified weights.')
            total_weight = len(templates)

        queries_per_weight = num_queries / total_weight
        train_output = list[QueryInstance[TQuery]]()
        val_output = list[QueryInstance[TQuery]]()

        for group in groups.values():
            train, val = self._generate_for_group(group, queries_per_weight)
            train_output.extend(train)
            val_output.extend(val)

        self._rng.shuffle(train_output)
        self._rng.shuffle(val_output)

        return train_output, val_output

    def _generate_for_group(self, group: list[CategorizedQueryTemplate[TQuery]], queries_per_weight: float) -> tuple[list[QueryInstance[TQuery]], list[QueryInstance[TQuery]]]:
        group_weight = sum(template.weight for template in group)

        if len(group) < 2 or group_weight == 0:
            print_warning(f'Category "{group[0].category}" has only {len(group)} definition(s) or zero weight. Cannot split into train/validation sets as intended. All queries from this category will be in the train set.')
            queries = self._generate_list(group, queries_per_weight)
            return queries, []

        shuffled = group[:]
        self._rng.shuffle(shuffled)

        train_weight = 0
        train_weight_target = group_weight * self._train_split
        index = 0
        for template in shuffled:
            index += 1
            train_weight += template.weight
            if train_weight >= train_weight_target:
                break

        if (index == len(shuffled)):
            # Ensure at least one definition goes to the validation set
            index -= 1
            train_weight -= shuffled[index].weight

        val_weight = group_weight - train_weight

        train_templates = shuffled[:index]
        val_templates = shuffled[index:]

        # Normalize train/validation weights to match the exact train_split as closely as possible
        num_queries = queries_per_weight * group_weight
        train_qpw = self._train_split * num_queries / train_weight
        val_qpw = (1 - self._train_split) * num_queries / val_weight

        train_queries = self._generate_list(train_templates, train_qpw)
        val_queries = self._generate_list(val_templates, val_qpw)

        return train_queries, val_queries

    def _generate_list(self, templates: list[CategorizedQueryTemplate[TQuery]], queries_per_weight: float) -> list[QueryInstance[TQuery]]:
        output = list[QueryInstance[TQuery]]()
        for template in templates:
            index = 0
            for _ in range(max(ceil(queries_per_weight * template.weight), 1)):
                query = template.generate(self._scale, index)
                output.append(query)
                index += 1

        return output

#endregion
