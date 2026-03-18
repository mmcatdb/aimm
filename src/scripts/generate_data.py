import argparse
from common.config import Config, DatasetName
from common.utils import exit_with_exception
from datasets.databases import get_available_dataset_names

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Generate database data.')

    common_args(parser)

    args = parser.parse_args(rawArgs)

    config = Config.load()
    dataset = DatasetName(args.dataset[0])
    import_directory = args.import_dir or config.dataset_import_directory(dataset)

    if dataset == DatasetName.EDBT:
        generate_edbt(config, import_directory, args.scale)
    elif dataset == DatasetName.TPCH:
        generate_tpch(config, import_directory, args.scale)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

def common_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset.')
    parser.add_argument('--import-dir', type=str, default=None, help='Path to the directory where the output files should be generated. If not specified, defaults to "{IMPORT_DIRECTORY}/<dataset>".')
    parser.add_argument('--scale', type=float, default=1, help='Scale factor for data generation. Default value (1.0) corresponds to ~100 MB so be responsible.')

def generate_tpch(config: Config, import_directory: str, scale: float):
    from datasets.tpch.data_generator import TpchDataGenerator

    try:
        generator = TpchDataGenerator(config)
        generator.run(import_directory, scale)
    except Exception as e:
        exit_with_exception(e)

def generate_edbt(config: Config, import_directory: str, scale: float):
    from datasets.edbt.data_generator import EdbtDataGenerator

    try:
        generator = EdbtDataGenerator(config)
        generator.run(import_directory, scale)
    except Exception as e:
        exit_with_exception(e)

if __name__ == '__main__':
    main()
