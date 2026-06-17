from __future__ import annotations

import argparse
import os

from core.config import Config
from core.drivers import DriverType
from core.dynamic_provider import get_dynamic_class_instance
from core.query import (
    QueryRegistry,
    load_measured,
    parse_database_id,
    parse_query_instance_id,
    parse_schema_id,
)
from core.utils import exit_with_exception
from providers.path_provider import PathProvider


SCHEMA = 'edbt'


def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Validate that EDBT measurement files contain all current registry templates.',
    )
    add_args(parser)
    args = parser.parse_args(raw_args)

    try:
        run(Config.load(), args)
    except Exception as exc:
        exit_with_exception(exc)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        'driver',
        choices=[driver_type.value for driver_type in DriverType],
        help='Database driver used by the measurement files.',
    )
    parser.add_argument(
        'measurement_suffixes',
        nargs='+',
        help='Measurement file suffixes. Pattern: edbt-{scale}/measured-{num_queries}-{num_runs}.jsonl',
    )


def run(config: Config, args: argparse.Namespace):
    driver_type = DriverType(args.driver)
    path_provider = PathProvider(config)

    for suffix in args.measurement_suffixes:
        schema, scale = _parse_edbt_measurement_suffix(suffix)
        registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
        expected_templates = _expected_template_names(registry, scale)
        path = path_provider.measured_by_suffix(driver_type, suffix)
        measured_templates = _measured_template_names(path, driver_type)

        missing_templates = sorted(expected_templates - measured_templates)
        if missing_templates:
            joined = ', '.join(missing_templates)
            raise ValueError(
                f'Measurement file {path!r} is missing current EDBT templates: {joined}. '
                'Regenerate measurements for this file before training.'
            )

        print(
            f'Validated {path}: '
            f'{len(expected_templates)} current EDBT templates present.'
        )


def _parse_edbt_measurement_suffix(suffix: str) -> tuple[str, float]:
    schema_id = suffix.split(os.sep, 1)[0].split('/', 1)[0]
    schema, scale = parse_schema_id(schema_id)
    if schema != SCHEMA:
        raise ValueError(
            f'Expected an {SCHEMA!r} measurement suffix, got {suffix!r}'
        )
    return schema, scale


def _expected_template_names(registry: QueryRegistry, scale: float) -> set[str]:
    return {
        parse_query_instance_id(query.id)[1]
        for query in registry.generate_queries(scale, 0, allow_write=True)
    }


def _measured_template_names(path: str, expected_driver_type: DriverType) -> set[str]:
    if not os.path.exists(path):
        raise ValueError(f'Measurement file {path!r} does not exist')

    measured = load_measured(path)
    driver_type, schema, _ = parse_database_id(measured.database_id)
    if driver_type != expected_driver_type:
        raise ValueError(
            f'Measurement file {path!r} belongs to driver {driver_type.value!r}, '
            f'expected {expected_driver_type.value!r}'
        )
    if schema != SCHEMA:
        raise ValueError(
            f'Measurement file {path!r} belongs to schema {schema!r}, expected {SCHEMA!r}'
        )

    return {
        parse_query_instance_id(item.id)[1]
        for item in measured.items
    }


if __name__ == '__main__':
    main()
