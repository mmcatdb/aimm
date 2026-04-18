import os
from core.config import Config
from core.query import DatabaseId, SchemaId
from latency_estimation.dataset import DatasetId
from latency_estimation.model import CheckpointId, ModelId
from latency_estimation.trainer import EPOCH_DIRECTORY

class PathProvider:
    def __init__(self, config: Config):
        self.config = config

    def imports(self, schema_id: SchemaId) -> str:
        return os.path.join(self.config.import_directory, schema_id)

    def populate_times(self, database_id: DatabaseId) -> str:
        return os.path.join(self.config.populate_directory, f'{database_id}.json')

    def _cache_dir(self, *parts: str) -> str:
        return os.path.join(self.config.cache_directory, *parts)

    def measured(self, database_id: DatabaseId, num_queries: int, num_runs: int) -> str:
        # Json lines!
        filename = f'measured-{num_queries}-{num_runs}.jsonl'
        return self._cache_dir(database_id, filename)

    def dataset(self, dataset_id: str) -> str:
        return self._cache_dir(dataset_id, 'dataset.pkl')

    def feature_extractor(self, dataset_id: DatasetId) -> str:
        return self._cache_dir(dataset_id, 'feature_extractor.pkl')

    def operators(self, dataset_id: DatasetId) -> str:
        return self._cache_dir(dataset_id, 'operators.json')

    MODEL_SUFFIX = '.pt'

    def model(self, checkpoint_id: CheckpointId) -> str:
        return os.path.join(self.config.checkpoints_directory, f'{checkpoint_id}{self.MODEL_SUFFIX}')

    METRICS_SUFFIX = '_metrics.json'

    def metrics(self, checkpoint_id: CheckpointId) -> str:
        return os.path.join(self.config.checkpoints_directory, f'{checkpoint_id}{self.METRICS_SUFFIX}')

    def epochs(self, model_id: ModelId) -> str:
        return os.path.join(self.config.checkpoints_directory, model_id, EPOCH_DIRECTORY)
