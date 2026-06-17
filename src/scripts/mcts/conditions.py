from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json

from search.mcts import AssignmentConditions, DatabaseInstance, WorkloadQuery


DatabaseRefResolver = Callable[[str], str]


def load_assignment_conditions(
    path: str | None,
    queries: Sequence[WorkloadQuery],
    databases: Sequence[DatabaseInstance],
    resolve_database_ref: DatabaseRefResolver | None = None,
) -> AssignmentConditions:
    if not path:
        return AssignmentConditions()

    try:
        with open(path, encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Assignment conditions file {path!r} is not valid JSON: {exc.msg}'
        ) from exc
    except OSError as exc:
        raise ValueError(f'Could not read assignment conditions file {path!r}: {exc}') from exc

    return parse_assignment_conditions(
        data,
        queries=queries,
        databases=databases,
        resolve_database_ref=resolve_database_ref,
    )


def parse_assignment_conditions(
    data: object,
    queries: Sequence[WorkloadQuery],
    databases: Sequence[DatabaseInstance],
    resolve_database_ref: DatabaseRefResolver | None = None,
) -> AssignmentConditions:
    if not isinstance(data, dict):
        raise ValueError('Assignment conditions file must contain a JSON object')

    allowed_keys = {'must_assign', 'must_not_assign'}
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            'Assignment conditions file contains unknown keys: '
            + ', '.join(repr(key) for key in unknown_keys)
        )

    resolver = resolve_database_ref or exact_database_ref_resolver(databases)
    query_ids = {query.id for query in queries}

    must_assign = _parse_must_assign(
        data.get('must_assign', {}),
        query_ids,
        resolver,
    )
    must_not_assign = _parse_must_not_assign(
        data.get('must_not_assign', {}),
        query_ids,
        resolver,
    )
    return AssignmentConditions(
        must_assign=must_assign,
        must_not_assign=must_not_assign,
    )


def exact_database_ref_resolver(
    databases: Sequence[DatabaseInstance],
) -> DatabaseRefResolver:
    database_ids = {database.id for database in databases}

    def resolve(database_ref: str) -> str:
        if database_ref not in database_ids:
            raise ValueError(
                f'Assignment condition references unknown database id {database_ref!r}'
            )
        return database_ref

    return resolve


def edbt_database_ref_resolver(
    databases: Sequence[DatabaseInstance],
) -> DatabaseRefResolver:
    exact_resolver = exact_database_ref_resolver(databases)
    database_id_by_driver = _database_id_by_driver(databases)

    def resolve(database_ref: str) -> str:
        if database_ref in database_id_by_driver:
            return database_id_by_driver[database_ref]
        return exact_resolver(database_ref)

    return resolve


def format_assignment_conditions(conditions: AssignmentConditions) -> list[str]:
    if conditions.is_empty:
        return []

    lines = ['Assignment conditions:']
    if conditions.must_assign:
        lines.append('  must assign:')
        for query_id in sorted(conditions.must_assign):
            lines.append(f'    {query_id} -> {conditions.must_assign[query_id]}')

    if conditions.must_not_assign:
        lines.append('  must not assign:')
        for query_id in sorted(conditions.must_not_assign):
            for database_id in sorted(conditions.must_not_assign[query_id]):
                lines.append(f'    {query_id} -> {database_id}')

    return lines


def print_assignment_conditions(conditions: AssignmentConditions):
    for line in format_assignment_conditions(conditions):
        print(line)


def assignment_conditions_allow(
    query_id: str,
    database_id: str,
    assignment_conditions: AssignmentConditions,
) -> bool:
    required_database_id = assignment_conditions.must_assign.get(query_id)
    if required_database_id is not None and database_id != required_database_id:
        return False
    return database_id not in assignment_conditions.must_not_assign.get(query_id, frozenset())


def _parse_must_assign(
    data: object,
    query_ids: set[str],
    resolve_database_ref: DatabaseRefResolver,
) -> dict[str, str]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError('Assignment conditions "must_assign" must be an object')

    output = {}
    for query_id, database_ref in data.items():
        _validate_query_id(query_id, query_ids)
        if not isinstance(database_ref, str):
            raise ValueError(
                f'Assignment condition database id for query {query_id!r} '
                f'must be a string: {database_ref!r}'
            )
        output[query_id] = resolve_database_ref(database_ref)
    return output


def _parse_must_not_assign(
    data: object,
    query_ids: set[str],
    resolve_database_ref: DatabaseRefResolver,
) -> dict[str, set[str]]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError('Assignment conditions "must_not_assign" must be an object')

    output = {}
    for query_id, database_refs in data.items():
        _validate_query_id(query_id, query_ids)
        if not isinstance(database_refs, list):
            raise ValueError(
                f'Assignment condition forbidden databases for query {query_id!r} '
                f'must be an array of strings'
            )

        normalized_database_ids = set()
        for database_ref in database_refs:
            if not isinstance(database_ref, str):
                raise ValueError(
                    f'Assignment condition forbidden database for query {query_id!r} '
                    f'must be a string: {database_ref!r}'
                )
            normalized_database_ids.add(resolve_database_ref(database_ref))
        output[query_id] = normalized_database_ids
    return output


def _validate_query_id(query_id: object, query_ids: set[str]):
    if not isinstance(query_id, str):
        raise ValueError(f'Assignment condition query id must be a string: {query_id!r}')
    if query_id not in query_ids:
        raise ValueError(f'Assignment condition references unknown query id {query_id!r}')


def _database_id_by_driver(databases: Sequence[DatabaseInstance]) -> Mapping[str, str]:
    from core.query import parse_database_id

    output = {}
    for database in databases:
        driver_type, _, _ = parse_database_id(database.id)
        output[driver_type.value] = database.id
    return output
