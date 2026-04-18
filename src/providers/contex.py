from core.config import Config
from core.driver_provider import DriverProvider
from latency_estimation.model_provider import ModelProvider
from providers.path_provider import PathProvider

class Context:
    """Wrapper for multiple providers to share the same configuration."""

    def __init__(self,
        config: Config,
        dp: DriverProvider,
        mp: ModelProvider,
        pp: PathProvider,
    ):
        self.config = config
        self.pp = pp
        self.dp = dp
        self.mp = mp

    @staticmethod
    def default(path: str | None = None, quiet: bool = False) -> 'Context':
        config = Config.load(path)
        pp = PathProvider(config)
        dp = DriverProvider.default(config)
        mp = ModelProvider(config, quiet)

        return Context(config, dp, mp, pp)
