from collections.abc import Callable
import csv
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from core.config import Config, GLOBAL_RNG_SEED
from core.query import SchemaName

class DataGenerator(ABC):
    """Base class for data generators."""

    # TODO Add back the config and use it for stuff like rng seed
    # def __init__(self, config: Config):
    #     self._config = config
    def __init__(self, schema: SchemaName):
        self._schema = schema

    def _reset(self, scale: float):
        self._scale = scale
        self._rng = random.Random(GLOBAL_RNG_SEED)
        self._now = datetime.now(timezone.utc)

    @abstractmethod
    def _generate_data(self) -> None:
        """Generates data at the specified scale and saves it to the import directory."""
        pass

    def run(self, scale: float, import_directory: str):
        self._reset(scale)
        self._import_directory = import_directory
        os.makedirs(self._import_directory, exist_ok=True)

        title = f'--- {self._schema} Data Generator ---'
        print(title)
        print(f'Scale factor: {self._scale:g}')
        print('-' * len(title) + '\n')

        self._create_generators()

        print(f'Generating data at scale factor {self._scale:g}...')
        self._generate_data()
        print(f'Data generation completed. Data saved to: {self._import_directory}')

    def _create_generators(self):
        # TODO This should be scaled ...
        self._rng_word = create_word_generator(self._rng, [ 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 11, 11, 12, 12 ], 10000)
        self._rng_name = create_word_generator(self._rng, [ 3, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10 ], 2000)
        self._rng_subdomain = create_word_generator(self._rng, [ 4, 5, 5, 6, 6, 7, 7, 8 ], 200)

        self._used_strings = set()
        """For emails and other unique fields."""

    def _open_csv_output(self, kind: str, header: list[str]):
        print('Creating', kind)
        path = os.path.join(self._import_directory, kind + '.tbl')
        file = open(path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(file, delimiter = '|')
        # The header is skipped because loaders do not expect it.
        # writer.writerow(header)
        return file, writer

    def _open_csv_input(self, kind: str, header: list[str]):
        print('Loading', kind)
        path = os.path.join(self._import_directory, kind + '.tbl')
        file = open(path, 'r', newline='', encoding='utf-8')
        reader = csv.reader(file, delimiter = '|')
        # The header is skipped because loaders do not expect it.
        # next(reader, None)
        return file, reader

    def _scaled(self, base: int, exp: float, min_v: int = 1) -> int:
        return max(min_v, int(round(base * (self._scale ** exp))))

    def _rng_timestamp_between(self, start: datetime, end: datetime) -> datetime:
        """Uniform timestamp between start and end."""
        delta = end - start
        seconds = int(delta.total_seconds())
        if seconds <= 0:
            return start
        return start + timedelta(seconds = self._rng.randint(0, seconds))

    def _rng_full_name(self) -> str:
        """Generates a random full name."""
        first_name = self._rng_name()
        last_name = self._rng_name()
        return f'{first_name} {last_name}'

    def _rng_unique_email(self, full_name: str) -> str:
        """Generates a random (but unique) email address based on the full name."""
        name_part = full_name.lower().replace(' ', '.')
        subdomain = self._rng_subdomain()

        email = f'{name_part}@{subdomain}.com'
        if email in self._used_strings:
            modifier = 1
            while email in self._used_strings:
                email = f'{name_part}{modifier}@{subdomain}.com'
                modifier += 1

        self._used_strings.add(email)
        return email

    def _rng_text(self, min_words: int, max_words: int) -> str:
        """Generates a random text with a word count between min_words and max_words."""
        word_count = self._rng.randint(min_words, max_words)
        words = [self._rng_word() for _ in range(word_count)]
        return ' '.join(words)

    def _rng_timestamp_since(self, years: int) -> datetime:
        """Uniform timestamp between the specified start (in years) now."""
        seconds = years * 365 * 24 * 60 * 60
        return self._now - timedelta(seconds = self._rng.randint(0, seconds))

    def _rng_date(self, start_year: int, end_year: int) -> datetime:
        """Generates a random date string between the specified years (inclusive on both ends)."""
        # NICE_TO_HAVE Kinda duplicated logic with `Database._rng_date`.
        years = end_year - start_year + 1
        # Not the most accurate way to handle leap years, but good enough for random generation.
        seconds = years * 365 * 24 * 60 * 60
        return datetime(start_year, 1, 1) + timedelta(seconds = self._rng.randint(0, seconds))

    def _rng_country_code(self) -> str:
        """Generates a random country code."""
        countries = ['US', 'GB', 'DE', 'FR', 'CZ', 'PL', 'ES', 'IT', 'NL', 'SE', 'IN', 'BR', 'CA', 'AU']
        return countries[self._rng.randrange(len(countries))]

    def _rng_time_zone(self) -> str:
        """Generates a random time zone."""
        time_zones = ['UTC', 'America/New_York', 'Europe/London', 'Europe/Berlin', 'Asia/Kolkata', 'America/Sao_Paulo']
        return time_zones[self._rng.randrange(len(time_zones))]

    def _rng_locale(self) -> str:
        """Generates a random locale."""
        locales = ['en_US', 'en_GB', 'de_DE', 'fr_FR', 'es_ES', 'it_IT', 'nl_NL', 'sv_SE', 'pl_PL', 'pt_BR']
        return locales[self._rng.randrange(len(locales))]

    def _rng_currency(self) -> str:
        """Generates a random currency code."""
        currencies = ['USD', 'EUR', 'GBP', 'INR', 'BRL', 'CAD', 'AUD']
        return currencies[self._rng.randrange(len(currencies))]

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

def create_word_generator(rng: random.Random, lengths: list[int], vocabulary_length: int, vocabulary: set[str] | None = None) -> Callable[[], str]:
    """Creates a word generator function that produces random words."""
    if vocabulary is None:
        vocabulary = set()

    vocabulary_list = list(vocabulary)

    def generate_word() -> str:
        if len(vocabulary_list) >= vocabulary_length:
            return rng.choice(vocabulary_list)

        while True:
            length = rng.choice(lengths)
            word = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=length))
            if word not in vocabulary:
                vocabulary.add(word)
                vocabulary_list.append(word)

            return word

    return generate_word
