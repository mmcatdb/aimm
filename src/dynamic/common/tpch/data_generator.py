import os
import shutil

from typing_extensions import override

from core.data_generator import DataGenerator


class TpchDataGenerator(DataGenerator):
    """Generates the repository's TPC-H extension data.

    The checked-out data already contains the standard TPC-H .tbl files. This
    generator copies those base files when available and regenerates the local
    `knows` relationship table.
    """

    def __init__(self):
        super().__init__('tpch')

    @override
    def _generate_data(self) -> None:
        self._copy_base_tables_if_available()
        customer_ids = self._load_customer_ids()
        self._generate_knows(customer_ids)

    def _copy_base_tables_if_available(self) -> None:
        source = os.path.join('data', 'inputs', 'tpch')
        if not os.path.isdir(source):
            return

        source = os.path.abspath(source)
        target = os.path.abspath(self._import_directory)
        if source == target:
            return

        for filename in os.listdir(source):
            if not filename.endswith('.tbl') or filename == 'knows.tbl':
                continue
            shutil.copyfile(os.path.join(source, filename), os.path.join(target, filename))

    def _load_customer_ids(self) -> list[int]:
        customer_ids: list[int] = []
        with self._open_csv_input('customer', [])[0] as file:
            for line in file:
                if not line.strip():
                    continue
                customer_ids.append(int(line.split('|', 1)[0]))
        return customer_ids

    def _generate_knows(self, customer_ids: list[int]) -> None:
        sources = ['organic', 'social_media', 'referral', 'advertisement', 'campaign', 'other']
        n_customers = len(customer_ids)
        n_knows = self._scaled(n_customers * 10, 1.0, min_v=min(n_customers, 1))

        file, writer = self._open_csv_output('knows', [])
        with file:
            written = 0
            while written < n_knows:
                from_id = customer_ids[self._rng.randrange(n_customers)]
                to_id = customer_ids[self._rng.randrange(n_customers)]
                if from_id == to_id:
                    continue

                writer.writerow([
                    from_id,
                    to_id,
                    self._rng_date(1992, 1998).strftime('%Y-%m-%d'),
                    self._rng.choice(sources),
                    self._rng_text(5, 16),
                    self._rng.random(),
                ])
                written += 1


def export():
    return TpchDataGenerator()
