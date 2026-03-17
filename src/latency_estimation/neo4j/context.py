import importlib
import os
import pickle
import sys
from common.config import Config, DatasetName
from common.drivers import DriverType, Neo4jDriver
from common.database import Database
from common.utils import print_warning
from datasets.databases import find_database, TRAIN_DATASET
from latency_estimation.context import BaseContext
from latency_estimation.common import load_checkpoint_file, load_or_create_dataset, save_checkpoint_file
from latency_estimation.neo4j.plan_extractor import PlanExtractor
from latency_estimation.neo4j.feature_extractor import FeatureExtractor
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork
from latency_estimation.neo4j.trainer import PlanStructuredTrainer

class Neo4jContext(BaseContext):
    def __init__(self, config: Config, quiet: bool):
        super().__init__(config, quiet)
        self.driver: Neo4jDriver
        self.database: Database[str]
        self.extractor: PlanExtractor

    @staticmethod
    def create(quiet: bool = False, dataset: DatasetName = TRAIN_DATASET) -> 'Neo4jContext':
        ctx = Neo4jContext(Config.load(), quiet)
        ctx.driver = Neo4jDriver(ctx.config.neo4j, dataset.value)
        ctx.database = find_database(dataset, DriverType.NEO4J)
        ctx.extractor = PlanExtractor(ctx.driver, ctx.database)
        return ctx

    def close(self):
        self.driver.close()

    def _load_feature_extractor_sidecar(self, checkpoint_path: str) -> FeatureExtractor | None:
        candidates = list(dict.fromkeys([
            os.path.join(os.path.dirname(checkpoint_path), 'neo4j_feature_extractor.pkl'),
            checkpoint_path.replace('_model.pt', '_feature_extractor.pkl'),
            checkpoint_path.replace('.pt', '_feature_extractor.pkl'),
        ]))

        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue

            previous_module = sys.modules.get('feature_extractor')
            try:
                sys.modules['feature_extractor'] = importlib.import_module('latency_estimation.neo4j.feature_extractor')

                with open(candidate, 'rb') as file:
                    extractor = pickle.load(file)

                if isinstance(extractor, FeatureExtractor):
                    if not self.quiet:
                        print(f'Loaded feature extractor from {candidate}')
                    return extractor
            except Exception as e:
                print_warning(f'Could not load Neo4j feature extractor from {candidate}.', e)
            finally:
                if previous_module is not None:
                    sys.modules['feature_extractor'] = previous_module
                else:
                    sys.modules.pop('feature_extractor', None)

        return None

    def load_or_create_dataset(self, num_queries: int, num_runs: int):
        return load_or_create_dataset(
            self._dataset_path(num_queries, num_runs),
            lambda: self.extractor.collect_training_dataset(num_queries, num_runs),
        )

    # These methods don't change the internal state of the context for a good reason - the use case is much more complex than it seems.
    # The trainer has the model. Sometimes we need only the model, sometimes both. But the trainer should not depend on the model - we might want to create a different model.
    # Similarly, the model shouldn't depend on a specific plan or feature extractor.
    # Each component should be responsible for what it saves to the checkpoint. However, in that case, it shouldn't depend on what is saved by other components (e.g., the trainer shouldn't instantiate the model from the checkpoint - it should be given the model).
    # So, we provide "stupid" methods that just save or load the components without any logic.

    def load_model(self, path: str | None) -> PlanStructuredNetwork:
        if not self.quiet:
            print('Loading trained model...')

        if not path:
            path = self._checkpoint_path('best')

        checkpoint = load_checkpoint_file(path, self.config.device)
        model_checkpoint = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model = PlanStructuredNetwork.from_checkpoint(model_checkpoint, self.config.device)

        if not (isinstance(model_checkpoint, dict) and model_checkpoint.get('feature_extractor') is not None):
            extractor = self._load_feature_extractor_sidecar(path)
            if extractor is not None:
                model.feature_extractor = extractor

        if not self.quiet:
            print('Model loaded successfully!\n')
            model.print_summary()
            if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
                PlanStructuredTrainer.print_metrics(checkpoint['metrics'])

        return model

    def save_checkpoint(self, suffix: str, model: PlanStructuredNetwork, trainer: PlanStructuredTrainer, metrics: dict[str, float]) -> None:
        path = self._checkpoint_path(suffix)

        save_checkpoint_file(path, {
            'model': model.to_checkpoint(),
            'trainer': trainer.to_checkpoint(),
            # Measured metrics (just so that we can see them easily).
            'metrics': metrics,
        }, not self.was_checkpoint_saved)

        self.was_checkpoint_saved = True

        if not self.quiet:
            print(f'Model saved to {path}')

    def load_checkpoint(self, path: str, learning_rate: float, batch_size: int) -> tuple[PlanStructuredNetwork, PlanStructuredTrainer]:
        if not self.quiet:
            print('Loading trained model and trainer...')

        checkpoint = load_checkpoint_file(path, self.config.device)
        model = PlanStructuredNetwork.from_checkpoint(checkpoint['model'], self.config.device)
        trainer = PlanStructuredTrainer.load_from_checkpoint(model, checkpoint['trainer'], learning_rate, batch_size)

        if not self.quiet:
            print('Model and trainer loaded successfully!\n')
            model.print_summary()
            PlanStructuredTrainer.print_metrics(checkpoint['metrics'])

        return model, trainer
