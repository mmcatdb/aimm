import json
import os
from abc import ABC
from common.config import Config
from common.utils import print_warning
from common.database import Database
from common.nn_operator import NnOperator
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork

class BaseContext(ABC):
    def __init__(self, config: Config, quiet: bool):
        self.config = config
        self.quiet = quiet
        self.database: Database

        self.was_checkpoint_saved = False

    def _dataset_path(self, num_queries: int, num_runs: int) -> str:
        return os.path.join(self.config.cache_directory, f'{self.database.id()}_{num_queries}_{num_runs}.pkl')

    def _checkpoint_path(self, suffix: str) -> str:
        return os.path.join(self.config.checkpoint_directory, f'{self.database.id()}_{suffix}.pt')

    @staticmethod
    def available_operators_path(config: Config, database: Database) -> str:
        # We don't cache the operators by the dataset because they are shared between different datasets.
        # I.e., the dataset is the one on which the operators were collected, but we might want to use them for a different dataset.
        # NICE_TO_HAVE This is not ideal, we might want to tie the operators to some checkpoint instead. But let's just do it like this for now.
        return os.path.join(config.cache_directory, f'{database.type.value}_operators.json')

    def save_available_operators(self, network: BasePlanStructuredNetwork):
        path = self.available_operators_path(self.config, self.database)

        try:
            with open(path, 'w') as file:
                sorted_operators = sorted(network.get_units(), key=lambda x: x.key())
                operators = [unit.to_dict() for unit in sorted_operators]
                json.dump(operators, file, indent=4)
        except Exception as e:
            print_warning(f'Could not save available operators to {path}.', e)

    # This is static because we need it in some other modules without instantiating the context.
    @staticmethod
    def load_available_operators(config: Config, database: Database) -> list[NnOperator] | None:
        path = BaseContext.available_operators_path(config, database)

        try:
            with open(path, 'r') as file:
                operators_data = json.load(file)
                return [NnOperator.from_dict(data) for data in operators_data]
        except FileNotFoundError:
            print_warning(f'No available operators file found at {path}.')
            return None
        except Exception as e:
            print_warning(f'Could not load available operators from {path}.', e)
            return None
