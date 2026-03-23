from typing_extensions import override
from common.data_generator import DataGenerator, clamp_int

class TpchDataGenerator(DataGenerator):
    """
    Data generator for the Tpch dataset.
    - Only adds some tables, doesn't change the original ones.

    Kinds: knows.
    """
    @override
    def name(self):
        """Returns the name of the generator (for display purposes)."""
        return 'TPC-H'

    @override
    def _generate_data(self):
        customer_ids = self._load_customer_ids()
        self._generate_knows(customer_ids)

    def _load_customer_ids(self) -> list[int]:
        f, r = self._open_csv_input('customer', ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'])
        with f:
            return [int(row[0]) for row in r]

    def _generate_knows(self, customer_ids: list[int]) -> None:
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
