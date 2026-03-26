from typing_extensions import override
from common.config import Config, DatasetName
from common.drivers import DriverType, MongoDriver
from common.database import DatabaseInfo
from latency_estimation.context import BaseContext
from latency_estimation.common import DatasetBundle, load_or_create_dataset
from latency_estimation.config import TrainConfig
from datasets.databases import TRAIN_DATASET
from latency_estimation.mongo.plan_extractor import PlanExtractor
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork
from latency_estimation.mongo.trainer import Trainer

class MongoContext(BaseContext[PlanStructuredNetwork]):
    def __init__(self, config: Config, quiet: bool):
        super().__init__(config, quiet)
        self.driver: MongoDriver
        self.extractor: PlanExtractor

    @staticmethod
    def create(quiet: bool = False, dataset: DatasetName = TRAIN_DATASET) -> 'MongoContext':
        ctx = MongoContext(Config.load(), quiet)
        ctx.driver = MongoDriver(ctx.config.mongo, dataset.value)
        ctx.info = DatabaseInfo(dataset, DriverType.MONGO)
        ctx.extractor = PlanExtractor(ctx.driver)
        return ctx

    def close(self):
        self.driver.close()

    def load_or_create_dataset(self, config: TrainConfig):
        return load_or_create_dataset(
            self._dataset_path(config.num_queries, config.num_runs),
            lambda: self.__create_dataset(config),
        )

    def __create_dataset(self, config: TrainConfig):
        def_map, train_queries, val_queries = self.database().generate_queries(config.num_queries, config.train_split)
        train = self.extractor.create_dataset(train_queries, config.num_runs, def_map=def_map)
        val = self.extractor.create_dataset(val_queries, config.num_runs, def_map=def_map)

        return DatasetBundle(train, val)

    @override
    def _create_model(self, checkpoint_model: dict) -> PlanStructuredNetwork:
        return PlanStructuredNetwork.from_checkpoint(checkpoint_model, self.config.device)

    @override
    def _create_trainer(self, checkpoint_trainer: dict, model: PlanStructuredNetwork, learning_rate: float, batch_size: int) -> Trainer:
        return Trainer.load_from_checkpoint(model, checkpoint_trainer, learning_rate, batch_size)
