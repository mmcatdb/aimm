import os
import argparse
from common.config import Config

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Clean cache')
    subparsers = parser.add_subparsers(dest='type', required=True)

    plans_args(subparsers.add_parser('plans'))

    args = parser.parse_args(rawArgs)

    if args.type == 'plans':
        plans_run(args)

def plans_args(parser: argparse.ArgumentParser):
    # Here can be something that specifies which cache to clean, e.g., dataset name, number of queries, etc.
    # parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset. Needed to select the database.')
    pass

def plans_run(args: argparse.Namespace):
    config = Config.load()
    cache_dir = config.cache_directory
    # for each .pkl file in cache_dir, delete the file:
    for filename in os.listdir(cache_dir):
        if not filename.endswith('.pkl'):
            continue

        path = os.path.join(cache_dir, filename)
        try:
            os.remove(path)
            print(f'Deleted cache file: {path}')
        except Exception as e:
            print(f'Error deleting file {path}: {e}')

if __name__ == '__main__':
    main()
