import os
from core.config import Config
from core.drivers import DriverType
from core.query import DatabaseId, MeasurementConfig, SchemaId
from latency_estimation.dataset import DatasetId
from latency_estimation.model import CheckpointId, ModelId
from latency_estimation.trainer import EPOCH_DIRECTORY

class PathProvider:
    def __init__(self, config: Config):
        self.config = config

    def imports(self, schema_id: SchemaId) -> str:
        return os.path.join(self.config.import_directory, schema_id)

    def _cache_dir(self, *parts: str) -> str:
        return os.path.join(self.config.cache_directory, *parts)

    def database_dir(self, database_id: DatabaseId, filename: str) -> str:
        return self._cache_dir(database_id, filename)

    def populate_times(self, database_id: DatabaseId) -> str:
        return self.database_dir(database_id, 'populate.json')

    def measured(self, database_id: DatabaseId, mc: MeasurementConfig) -> str:
        nowrite_suffix = '' if mc.allow_write else '-nowrite'
        # Json lines!
        filename = f'measured-{mc.num_queries}-{mc.num_runs}{nowrite_suffix}.jsonl'
        return self.database_dir(database_id, filename)

    def measured_by_suffix(self, driver_type: DriverType, suffix: str) -> str:
        """Suffix pattern: {schema_id}/measured-{num_queries}-{num_runs}.jsonl"""
        return self._cache_dir(driver_type.value, suffix)

    def dataset(self, dataset_id: str) -> str:
        return self._cache_dir(dataset_id, 'dataset.pkl')

    def flat_dataset(self, dataset_id: str) -> str:
        return self._cache_dir(dataset_id, 'flat_dataset.pkl')

    def feature_extractor(self, dataset_id: DatasetId) -> str:
        return self._cache_dir(dataset_id, 'feature_extractor.pkl')

    def flat_feature_extractor(self, dataset_id: DatasetId) -> str:
        return self._cache_dir(dataset_id, 'flat_feature_extractor.pkl')

    def operators(self, dataset_id: DatasetId) -> str:
        return self._cache_dir(dataset_id, 'operators.json')

    MODEL_SUFFIX = '.pt'

    def model(self, checkpoint_id: CheckpointId) -> str:
        return os.path.join(self.config.checkpoints_directory, f'{checkpoint_id}{self.MODEL_SUFFIX}')

    def flat_model(self, model_id: ModelId) -> str:
        return os.path.join(self.config.checkpoints_directory, model_id, 'flat_model.pkl')

    METRICS_SUFFIX = '_metrics.json'

    def metrics(self, checkpoint_id: CheckpointId) -> str:
        return os.path.join(self.config.checkpoints_directory, f'{checkpoint_id}{self.METRICS_SUFFIX}')

    def flat_metrics(self, model_id: ModelId) -> str:
        return os.path.join(self.config.checkpoints_directory, model_id, 'flat_metrics.json')

    def epochs(self, model_id: ModelId) -> str:
        return os.path.join(self.config.checkpoints_directory, model_id, EPOCH_DIRECTORY)
