from common.config import Config
from common.drivers import PostgresDriver
from datasets.tpch.postgres_database import TpchPostgresDatabase
from latency_estimation.common import load_checkpoint_file, load_dataset, save_checkpoint_file
from latency_estimation.postgres.plan_extractor import PlanExtractor
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork
from latency_estimation.postgres.trainer import PlanStructuredTrainer

class Context:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.config: Config
        self.postgres: PostgresDriver
        self.database: TpchPostgresDatabase
        self.extractor: PlanExtractor

    @staticmethod
    def create(quiet: bool = False) -> 'Context':
        ctx = Context(quiet)
        ctx.quiet = quiet
        ctx.config = Config.load()
        ctx.postgres = PostgresDriver(ctx.config.postgres)
        ctx.database = TpchPostgresDatabase()
        ctx.extractor = PlanExtractor(ctx.postgres, ctx.database)
        return ctx

    def close(self):
        self.postgres.close()

    def load_dataset(self, num_queries: int):
        cache_path = f'{self.config.cache_directory}/{self.database.id()}_{num_queries}.pkl'

        return load_dataset(cache_path, lambda: self.extractor.collect_training_dataset(num_queries))

    # These methods don't change the internal state of the context for a good reason - the use case is much more complex than it seems.
    # The trainer has the model. Sometimes we need only the model, sometimes both. But the trainer should not depend on the model - we might want to create a different model.
    # Similarly, the model shouldn't depend on a specific plan or feature extractor.
    # Each component should be responsible for what it saves to the checkpoint. However, in that case, it shouldn't depend on what is saved by other components (e.g., the trainer shouldn't instantiate the model from the checkpoint - it should be given the model).
    # So, we provide "stupid" methods that just save or load the components without any logic.

    def checkpoint_path(self, suffix: str) -> str:
        return f'{self.config.checkpoint_directory}/{self.database.id()}_{suffix}.pt'

    def load_model(self, path: str) -> PlanStructuredNetwork:
        if not self.quiet:
            print('Loading trained model...')

        checkpoint = load_checkpoint_file(path, self.config.device)
        model = PlanStructuredNetwork.from_checkpoint(checkpoint['model'], self.config.device)

        if not self.quiet:
            print('Model loaded successfully!\n')
            model.print_summary()
            PlanStructuredTrainer.print_metrics(checkpoint["metrics"])

        return model

    def save_checkpoint(self, suffix: str, model: PlanStructuredNetwork, trainer: PlanStructuredTrainer, metrics: dict[str, float]) -> None:
        path = self.checkpoint_path(suffix)

        save_checkpoint_file(path, {
            'model': model.to_checkpoint(),
            'trainer': trainer.to_checkpoint(),
            # Measured metrics (just so that we can see them easily).
            'metrics': metrics,
        })

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
            PlanStructuredTrainer.print_metrics(checkpoint["metrics"])

        return model, trainer
