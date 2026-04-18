import argparse
from core.config import Config
from core.utils import exit_with_exception
from latency_estimation.config import ModelConfig, TrainerConfig
from latency_estimation.dataset import create_dataset_id, load_dataset, parse_dataset_id, prune_dataset
from latency_estimation.model_provider import ModelProvider
from latency_estimation.plan_structured_network import ModelId, OperatorCollector
from ..providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Train a new QPP-Net model on a dataset.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    model_id = args.model_id[0]

    try:
        run(config, args, model_id)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id',        nargs=1,                 help='Id of the model. Pattern: {driver_type}/{model_name}.')
    parser.add_argument('..train-dataset', type=str, required=True, help='Name of the training dataset.')
    parser.add_argument('..val-dataset',   type=str, required=True, help='Name of the validation dataset.')

    parser.add_argument('--dry-run',       action='store_true',     help='Only print statistics about the dataset.')

    ModelConfig.add_arguments(parser)
    TrainerConfig.add_arguments(parser)

def run(config: Config, args: argparse.Namespace, model_id: ModelId):
    pp = PathProvider(config)

    driver_type, model_name = parse_dataset_id(model_id)

    model_config = ModelConfig.from_arguments(config, args, driver_type)
    print(f'Model config:')
    print(model_config)

    trainer_config = TrainerConfig.from_arguments(config, args, driver_type)
    print(f'Trainer config:')
    print(trainer_config)

    train_id = create_dataset_id(driver_type, args.train_dataset)
    train_dataset = load_dataset(pp.dataset(train_id))

    val_id = create_dataset_id(driver_type, args.val_dataset)
    val_dataset = load_dataset(pp.dataset(val_id))

    print(f'Training set: {len(train_dataset)} items')
    print(f'Validation set (original): {len(val_dataset)} items')

    print('\nCreating plan-structured neural network...')

    mp = ModelProvider(config)

    operators = OperatorCollector.run([item.plan for item in train_dataset])

    model = mp.create_model(driver_type, model_config, model_name, operators)
    model.print_summary()

    val_dataset = prune_dataset(val_dataset, model)
    print(f'Validation set (pruned): {len(val_dataset)} items')

    trainer = mp.create_trainer(model, trainer_config)

    if args.dry_run:
        print('\nDry run completed. Exiting before training.')
        return

    print(f'\nTraining for {trainer_config.num_epochs} epochs...')

    trainer.train_epochs(train_dataset, val_dataset, trainer_config.num_epochs, pp)
