from datetime import datetime, timezone
import os
import subprocess
from typing_extensions import override
from core.data_generator import DataGenerator, clamp_int
from core.query import SchemaName

def export():
    return TpchDataGenerator()

class TpchDataGenerator(DataGenerator):
    """
    Data generator for the Tpch schema.
    - Copies missing base TPC-H tables from data/inputs/tpch into the target
      scale directory.
    - Adds custom generated tables.

    Kinds: knows.
    """

    def __init__(self):
        super().__init__(self.SCHEMA, self.NOW)

    SCHEMA: SchemaName = 'tpch'
    # All generated timestamps are between 1992-01-01 and 1998-12-31. Specific fields have more strict boundaries but they all fall within this range.
    NOW = datetime(1999, 1, 1, tzinfo=timezone.utc)

    @override
    def _generate_data(self):
        self.__generate_tables()
        customer_ids = self.__load_customer_ids()
        self.__generate_knows(customer_ids)

    def __generate_tables(self):
        cmd: list[str] = [
            'tpchgen-cli',
            '--scale-factor', str(self.__get_tpch_scale()),
            '--format', 'csv',
            '--output-dir', self._import_directory,
        ]

        # Use at most half the available cores to avoid overloading the system.
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            cmd.append('--num-threads')
            cmd.append(str(cpu_count // 2))

        print(f'Running "{" ".join(cmd)}"')
        subprocess.run(cmd, check=True)

    def __get_tpch_scale(self) -> float:
        # The tpchgen-cli uses linear scale (10 times the scale is 10 times the data size). Approximate data sizes:
        # 0.1 -> 0.1 GB
        # 1 -> 1.1 GB
        # 10 -> 11 GB
        # So, we need to transform it from our logarithmic scale. Let's set 0 [our] as 0.1 [tpch]. Then, each increase of 1 in our scale should multiply the data size by 2.
        # Let's also round it to 2 significant digits for cleaner output.
        return round(0.1 * (2 ** self._scale), 2)

    def __load_customer_ids(self) -> list[int]:
        f, r = self._open_csv_input('customer', ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'])
        with f:
            return [int(row[0]) for row in r]

    def __generate_knows(self, customer_ids: list[int]) -> None:
        f, w = self._open_csv_output('knows', ['k_custkey1', 'k_custkey2', 'k_startdate', 'k_source', 'k_comment', 'k_strength'])

        sources = ['organic', 'social_media', 'referral', 'advertisement', 'campaign', 'other']

        n_customers = len(customer_ids)
        # On average, each customer knows 10 others. Scale with a bit more than linear to increase density.
        # There are 15k customers at scale factor 0 (we can't just use the number of loaded customers here because then it would be scaled twice).
        n_knows = self._scaled(15_000 * 10, 1.10)
        # Times 2 to compensate for the randomness. It will work out in average.
        k_coefficient = 2 * n_knows / n_customers

        used_pairs = set[tuple[int, int]]()

        from_index = 0
        while len(used_pairs) < n_knows:
            k = clamp_int(int(k_coefficient * self._rng.random()), 0, 25)
            remaining = n_knows - len(used_pairs)
            if k > remaining:
                k = remaining

            k_used = 0
            while k_used < k:
                to_index = self._rng.randint(0, n_customers - 1)
                sorted_pair = (from_index, to_index) if from_index < to_index else (to_index, from_index)
                if to_index == from_index or sorted_pair in used_pairs:
                    continue

                used_pairs.add(sorted_pair)
                k_used += 1

                w.writerow([
                    customer_ids[from_index],
                    customer_ids[to_index],
                    self._rng_date(1992, 1998).strftime('%Y-%m-%d'),
                    self._rng.choice(sources),
                    self._rng_text(5, 20),
                    f'{self._rng.random():.3f}',
                ])

            from_index = (from_index + 1) % n_customers

        f.close()
