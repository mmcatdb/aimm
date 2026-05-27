import os
import subprocess
from typing_extensions import override
from core.data_generator import DataGenerator, clamp_int

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
        super().__init__('tpch')

    @override
    def _generate_data(self):
        self.__generate_tables()
        customer_ids = self.__load_customer_ids()
        self.__generate_knows(customer_ids)

    def __generate_tables(self):
        cmd: list[str] = [
            'tpchgen-cli',
            '--scale-factor', str(self.__get_tpch_scale()),
            '--format', 'tbl',
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
        n_knows = self._scaled(n_customers * 10, 1.10)

        knows_written = 0
        from_index = 0
        while knows_written < n_knows:
            # Each person knows a small number
            k = clamp_int(int((self._rng.random() ** 1.2) * 25), 0, 25)
            if knows_written + k > n_knows:
                k = n_knows - knows_written

            chosen = set()
            while len(chosen) < k:
                to_index = self._rng.randint(0, n_customers - 1)
                if to_index != from_index:
                    chosen.add(to_index)

            for to_index in chosen:
                w.writerow([
                    customer_ids[from_index],
                    customer_ids[to_index],
                    self._rng_date(1992, 1998).strftime('%Y-%m-%d'),
                    self._rng.choice(sources),
                    self._rng_text(5, 20),
                    self._rng.random() * 1.00
                ])
                knows_written += 1
                if knows_written >= n_knows:
                    break

            from_index = (from_index + 1) % n_customers

        f.close()
