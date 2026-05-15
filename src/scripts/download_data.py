import argparse
import json
import os
import urllib.request
from core.config import Config
from core.drivers import DriverType
from core.query import DatabaseId, SchemaId, create_database_id_1, parse_schema_id, print_warning
from core.utils import ProgressTracker, exit_with_error, exit_with_exception, plural
from providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Downloads measurement data from a server and saves it to the cache.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()

    try:
        run(config, args.filter, args.keep_local)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('filter',       nargs='*',           help='If provided, only databases matching at least one filter will be considered. Pattern: {driver_type}/{schema_name}-{scale} (parts can be omitted from the right). Example: `postgres`, `postgres/art`, `postgres/art-0`.')
    parser.add_argument('--keep-local', action='store_true', help='If a local cache file already exists, it will be kept (even if the file is incomplete).')

def run(config: Config, filter: list[str], keep_local: bool):
    try:
        matcher = Matcher.create(filter)
    except ValueError as e:
        exit_with_error(str(e))


    root_url = config.download_data_url
    if not root_url:
        exit_with_error('No DOWNLOAD_DATA_URL provided in config.')

    progress = ProgressTracker.unlimited()
    ctx = Context(matcher, config, keep_local, progress)

    progress.start('Processing files... ')
    process_root(ctx, root_url)
    progress.finish()

    print()

    if ctx.written_files:
        print(f'Downloaded {plural(len(ctx.written_files), "new file")}:')
        for filename in ctx.written_files:
            print(f'  {filename}')

    if ctx.overwritten_files:
        print(f'Overwrote {plural(len(ctx.overwritten_files), "existing file")}:')
        for filename in ctx.overwritten_files:
            print(f'  {filename}')

    if ctx.skipped_files:
        print(f'Skipped {plural(len(ctx.skipped_files), "existing file")}:')
        for filename in ctx.skipped_files:
            print(f'  {filename}')

def process_root(ctx: Context, root_url: str):
    with urllib.request.urlopen(root_url) as response:
        data = json.loads(response.read())

    for item in data:
        try:
            driver = DriverType(item['name'])
            if item['type'] == 'directory' and ctx.matcher.matches(driver, None):
                driver_url = root_url + '/' + driver.value
                process_driver(ctx, driver_url, driver)
        except Exception as e:
            print_warning(f'Unexpected exception for driver. Data:\n{json.dumps(item)}.', e)
            pass

def process_driver(ctx: Context, driver_url: str, driver: DriverType):
    with urllib.request.urlopen(driver_url) as response:
        data = json.loads(response.read())

    for item in data:
        try:
            schema_id = item['name']
            parse_schema_id(schema_id) # Just to validate the schema id.
            if item['type'] == 'directory' and ctx.matcher.matches(driver, schema_id):
                database_url = driver_url + '/' + schema_id
                database_id = create_database_id_1(driver, schema_id)
                process_database(ctx, database_url, database_id)
        except Exception as e:
            print_warning(f'Unexpected exception for database. Data:\n{json.dumps(item)}.', e)
            pass

def process_database(ctx: Context, database_url: str, database_id: DatabaseId):
    with urllib.request.urlopen(database_url) as response:
        data = json.loads(response.read())

    for item in data:
        filename = None
        try:
            if item['type'] != 'file':
                continue

            filename = item['name']
            file_url = database_url + '/' + filename
            local_path = ctx.pp.database_dir(database_id, filename)
            display_filename = f'{database_id}/{filename}'

            already_exists = os.path.exists(local_path)
            if already_exists and ctx.keep_local:
                ctx.skipped_files.append(display_filename)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                urllib.request.urlretrieve(file_url, local_path)
                if already_exists:
                    ctx.overwritten_files.append(display_filename)
                else:
                    ctx.written_files.append(display_filename)

        except Exception as e:
            print_warning(f'Could not download file "{filename}" for database "{database_id}". Data:\n{json.dumps(item)}.', e)
            pass

        ctx.progress.track()

class Context:
    def __init__(self, matcher: Matcher, config: Config, keep_local: bool, progress: ProgressTracker):
        self.matcher = matcher
        self.config = config
        self.keep_local = keep_local
        self.pp = PathProvider(config)
        self.written_files = list[str]()
        self.overwritten_files = list[str]()
        self.skipped_files = list[str]()
        self.progress = progress

MatcherMap = dict
"""A dict (by driver type) of list of dicts (by schema) of set (of scales). Too long to write so you just gotta believe me."""

class Matcher:
    def __init__(self, map: MatcherMap):
        self.map = map

    @staticmethod
    def create(filter: list[str]):
        map: MatcherMap = {}
        for item in filter:
            try:
                driver, schema_name, scale = Matcher._parse_item(item)
                map.setdefault(driver, {}).setdefault(schema_name, set()).add(scale)
            except ValueError as e:
                raise ValueError(f'Invalid filter value in "{item}": {e}.')

        return Matcher(map)

    @staticmethod
    def _parse_item(item: str) -> tuple[DriverType, str | None, float | None]:
        by_slash = item.split('/')
        if len(by_slash) > 2:
            raise ValueError(f'invalid database_id: "{item}"')

        driver_type_str = by_slash[0]
        try:
            driver = DriverType(driver_type_str)
        except ValueError:
            raise ValueError(f'invalid driver type: "{driver_type_str}"')

        if len(by_slash) == 1:
            return driver, None, None

        schema_id = by_slash[1]
        by_dash = schema_id.split('-')
        if len(by_dash) > 2:
            raise ValueError(f'invalid schema_id: "{schema_id}"')

        schema_name = by_dash[0]
        if len(by_dash) == 1:
            return driver, schema_name, None

        scale_str = by_dash[1]
        try:
            scale = float(scale_str)
        except ValueError:
            raise ValueError(f'invalid scale: "{scale_str}"')

        return driver, schema_name, scale

    def matches(self, driver: DriverType, schema_id: SchemaId | None) -> bool:
        if not self.map:
            return True

        if driver not in self.map:
            return False

        if schema_id is None:
            return True

        by_schema = self.map[driver]
        if not by_schema:
            return True

        schema_name, scale = parse_schema_id(schema_id)

        if schema_name not in by_schema:
            return False

        by_scale = by_schema[schema_name]
        if not by_scale:
            return True

        return scale in by_scale

if __name__ == '__main__':
    main()
