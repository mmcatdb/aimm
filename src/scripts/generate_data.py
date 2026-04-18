import argparse
from core.config import Config
from core.data_generator import DataGenerator
from core.query import parse_schema_id
from core.utils import exit_with_exception
from core.dynamic_provider import get_dynamic_class_instance
from ..providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Generate schema data.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    schema_id = args.schema_id[0]

    try:
        run(config, schema_id)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('schema_id', nargs=1, help=f'Id of the schema. Pattern: {{schema_name}}-{{scale}}')

def run(config: Config, schema_id: str):
    pp = PathProvider(config)
    schema, scale = parse_schema_id(schema_id)

    generator = get_dynamic_class_instance(DataGenerator, None, schema)
    generator.run(scale, pp.imports(schema_id))

if __name__ == '__main__':
    main()
