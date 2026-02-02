import csv
import os
import random
import argparse
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from common.config import Config

class DataGenerator(ABC):
    """Base class for data generators."""
    def __init__(self, config: Config):
        self._config = config
        self._rng = random.Random(datetime.now().timestamp())
        self._now = datetime.now(timezone.utc)

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the generator (for display purposes)."""
        pass

    @abstractmethod
    def _generate_data(self) -> None:
        """Generates data at the specified scale and saves it to the data directory."""
        pass

    def run(self):
        args = self._parse_args()

        self._data_directory = args.data_dir or self._config.import_directory
        self._scale = args.scale

        title = f'--- {self.name()} Data Generator ---'
        print(title)
        print(f'Scale factor: {self._scale}')
        print('-' * len(title) + '\n')

        try:
            print(f'Generating data at scale factor {self._scale}...')
            self._generate_data()
            print(f'Data generation complete. Data saved to: {self._data_directory}')
        except Exception as e:
            print(f'\nError: {e}')

    def _parse_args(self):
        parser = argparse.ArgumentParser(description=f'Generate {self.name()} data.')
        parser.add_argument(
            '--data-dir',
            type=str,
            default=None,
            help=f'Path to the output directory for the {self.name()} .tbl files. If not specified, reads from "IMPORT_DIRECTORY" in .env.'
        )
        parser.add_argument(
            '--scale',
            type=float,
            default=1,
            help='Scale factor for data generation. Default value (1.0) corresponds to ~100 MB so be responsible.'
        )

        return parser.parse_args()

    def _open_csv(self, kind: str, header: list[str]):
        print('Creating', kind + '.tbl')
        filename = kind + '.tbl'
        path = os.path.join(self._data_directory, filename)
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f, delimiter = '|')
        w.writerow(header)
        return f, w

    def _scaled(self, base: int, exp: float, min_v: int = 1) -> int:
        return max(min_v, int(round(base * (self._scale ** exp))))

    def _rand_ts_between(self, start: datetime, end: datetime) -> datetime:
        """Uniform timestamp between start and end."""
        delta = end - start
        seconds = int(delta.total_seconds())
        if seconds <= 0:
            return start
        return start + timedelta(seconds = self._rng.randint(0, seconds))

    def _rand_ts_since(self, years: int) -> datetime:
        """Uniform timestamp between the specified start (in years) now."""
        seconds = years * 365 * 24 * 60 * 60
        return self._now - timedelta(seconds = self._rng.randint(0, seconds))

    def _weighted_choice_int(self, weights: list[float]) -> int:
        """Simple helper for small lists."""
        total = sum(weights)
        r = self._rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i

        return len(weights) - 1

    def _create_sampler(self, weights: list[float]) -> 'AliasSampler':
        return AliasSampler(self._rng, weights)

def iso(datetime: datetime) -> str:
    return datetime.isoformat()

def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

class AliasSampler:
    """O(1) discrete sampling after O(n) build. Good for many picks from a fixed weight list."""
    def __init__(self, rng: random.Random, weights: list[float]):
        self._rng = rng
        n = len(weights)
        if n == 0:
            raise ValueError("weights must not be empty")
        s = sum(weights)
        if s <= 0:
            raise ValueError("sum(weights) must be > 0")

        scaled = [w * n / s for w in weights]
        small = []
        large = []
        for i, x in enumerate(scaled):
            (small if x < 1.0 else large).append(i)

        self.prob = [0.0] * n
        self.alias = [0] * n

        while small and large:
            s_i = small.pop()
            l_i = large.pop()
            self.prob[s_i] = scaled[s_i]
            self.alias[s_i] = l_i
            scaled[l_i] = scaled[l_i] - (1.0 - scaled[s_i])
            (small if scaled[l_i] < 1.0 else large).append(l_i)

        for i in large + small:
            self.prob[i] = 1.0
            self.alias[i] = i

    def sample_index(self) -> int:
        n = len(self.prob)
        i = self._rng.randrange(n)
        if self._rng.random() < self.prob[i]:
            return i
        return self.alias[i]
