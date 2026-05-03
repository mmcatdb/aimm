import json

from typing_extensions import override

from core.data_generator import DataGenerator, iso


class EdbtDataGenerator(DataGenerator):
    """Generates synthetic e-commerce/social data for the EDBT schema."""

    def __init__(self):
        super().__init__('edbt')

    @override
    def _generate_data(self) -> None:
        counts = self._generate_counts()
        persons = self._generate_persons(counts['person'])
        sellers = self._generate_sellers(counts['seller'])
        products = self._generate_products(counts['product'], sellers)
        self._generate_categories(counts['category'])
        self._generate_has_category(counts['has_category'], counts['product'], counts['category'])
        self._generate_has_interest(counts['has_interest'], counts['person'], counts['category'])
        self._generate_follows(counts['follows'], counts['person'])
        self._generate_customers_orders_items(counts['customer'], counts['order'], counts['order_item'], persons, products)
        self._generate_reviews(counts['review'], counts['product'], counts['customer'])

    def _generate_counts(self) -> dict[str, int]:
        return {
            'person': self._scaled(50_000, 1.0),
            'seller': self._scaled(5_000, 1.0),
            'category': self._scaled(2_000, 0.7),
            'product': self._scaled(20_000, 1.0),
            'customer': self._scaled(200_000, 1.0),
            'order': self._scaled(200_000, 1.0),
            'order_item': self._scaled(310_000, 1.0),
            'review': self._scaled(100_000, 1.0),
            'has_category': self._scaled(30_000, 1.0),
            'has_interest': self._scaled(240_000, 1.0),
            'follows': self._scaled(200_000, 1.0),
        }

    def _profile_json(self) -> str:
        return json.dumps({
            'bio': self._rng_text(5, 20),
            'tz': self._rng_time_zone(),
            'lang': self._rng_locale(),
        }, ensure_ascii=True)

    def _generate_persons(self, n_persons: int) -> list[tuple[str, str, str, str, str, str]]:
        people: list[tuple[str, str, str, str, str, str]] = []
        file, writer = self._open_csv_output('person', [])
        with file:
            for person_id in range(1, n_persons + 1):
                name = self._rng_full_name()
                row = (
                    name,
                    self._rng_unique_email(name),
                    iso(self._rng_timestamp_since(5)),
                    self._rng_country_code(),
                    'true' if self._rng.random() > 0.04 else 'false',
                    self._profile_json(),
                )
                people.append(row)
                writer.writerow([person_id, *row])
        return people

    def _generate_sellers(self, n_sellers: int) -> list[int]:
        seller_ids: list[int] = []
        file, writer = self._open_csv_output('seller', [])
        with file:
            for seller_id in range(1, n_sellers + 1):
                seller_ids.append(seller_id)
                writer.writerow([
                    seller_id,
                    self._rng_full_name(),
                    iso(self._rng_timestamp_since(6)),
                    self._rng_country_code(),
                    'true' if self._rng.random() > 0.08 else 'false',
                ])
        return seller_ids

    def _generate_categories(self, n_categories: int) -> None:
        file, writer = self._open_csv_output('category', [])
        with file:
            for category_id in range(1, n_categories + 1):
                name = self._rng_word()
                root = max(1, int(category_id ** 0.5))
                writer.writerow([category_id, name, f'/cat{root}/{name}{category_id}'])

    def _generate_products(self, n_products: int, seller_ids: list[int]) -> dict[int, tuple[int, str, int, str]]:
        products: dict[int, tuple[int, str, int, str]] = {}
        colors = ['black', 'white', 'blue', 'green', 'red', 'silver']
        file, writer = self._open_csv_output('product', [])
        with file:
            for product_id in range(1, n_products + 1):
                seller_id = self._rng.choice(seller_ids)
                title = self._rng_text(3, 7)
                price = self._rng.randint(100, 200_000)
                currency = self._rng_currency()
                created_at = self._rng_timestamp_since(6)
                attrs = {
                    'brand': self._rng_word(),
                    'color': self._rng.choice(colors),
                    'weight_g': self._rng.randint(20, 20_000),
                }
                products[product_id] = (price, currency, title, seller_id)
                writer.writerow([
                    product_id,
                    seller_id,
                    f'SKU-{product_id:09d}',
                    title,
                    self._rng_text(8, 24),
                    price,
                    currency,
                    self._rng.randint(0, 5000),
                    'true' if self._rng.random() > 0.03 else 'false',
                    iso(created_at),
                    iso(created_at),
                    json.dumps(attrs, ensure_ascii=True),
                ])
        return products

    def _generate_has_category(self, n_rows: int, n_products: int, n_categories: int) -> None:
        seen: set[tuple[int, int]] = set()
        file, writer = self._open_csv_output('has_category', [])
        with file:
            while len(seen) < n_rows:
                key = (self._rng.randint(1, n_products), self._rng.randint(1, n_categories))
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow([key[0], key[1], iso(self._rng_timestamp_since(4))])

    def _generate_has_interest(self, n_rows: int, n_persons: int, n_categories: int) -> None:
        seen: set[tuple[int, int]] = set()
        file, writer = self._open_csv_output('has_interest', [])
        with file:
            while len(seen) < n_rows:
                key = (self._rng.randint(1, n_persons), self._rng.randint(1, n_categories))
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow([key[0], key[1], self._rng.randint(1, 10), iso(self._rng_timestamp_since(4))])

    def _generate_follows(self, n_rows: int, n_persons: int) -> None:
        seen: set[tuple[int, int]] = set()
        file, writer = self._open_csv_output('follows', [])
        with file:
            while len(seen) < n_rows:
                key = (self._rng.randint(1, n_persons), self._rng.randint(1, n_persons))
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                writer.writerow([key[0], key[1], iso(self._rng_timestamp_since(5))])

    def _generate_customers_orders_items(
        self,
        n_customers: int,
        n_orders: int,
        n_items: int,
        people: list[tuple[str, str, str, str, str, str]],
        products: dict[int, tuple[int, str, str, int]],
    ) -> None:
        customer_file, customer_writer = self._open_csv_output('customer', [])
        order_file, order_writer = self._open_csv_output('order', [])
        item_file, item_writer = self._open_csv_output('order_item', [])

        statuses = ['paid', 'shipped', 'canceled', 'refunded']
        product_ids = list(products.keys())

        with customer_file, order_file, item_file:
            for customer_id in range(1, n_customers + 1):
                person_id = self._rng.randint(1, len(people))
                person = people[person_id - 1]
                full_name, email, _, country_code, is_active, profile = person
                customer_writer.writerow([
                    customer_id,
                    person_id,
                    iso(self._rng_timestamp_since(2)),
                    full_name,
                    email,
                    country_code,
                    is_active,
                    profile,
                ])

            item_id = 1
            for order_id in range(1, n_orders + 1):
                customer_id = self._rng.randint(1, n_customers)
                ordered_at = iso(self._rng_timestamp_since(2))
                target_items = max(1, round(n_items / n_orders))
                if item_id + target_items > n_items + 1:
                    target_items = n_items - item_id + 1

                order_total = 0
                rows = []
                for _ in range(target_items):
                    product_id = self._rng.choice(product_ids)
                    price, currency, title, _ = products[product_id]
                    quantity = self._rng.randint(1, 4)
                    line_total = price * quantity
                    order_total += line_total
                    rows.append([item_id, order_id, product_id, price, quantity, line_total, ordered_at, json.dumps({
                        'title': title,
                        'price_cents': price,
                        'currency': currency,
                    }, ensure_ascii=True)])
                    item_id += 1

                order_writer.writerow([
                    order_id,
                    customer_id,
                    ordered_at,
                    self._rng.choice(statuses),
                    order_total,
                    rows[0][7] and products[rows[0][2]][1],
                    json.dumps({'method': self._rng.choice(['standard', 'express']), 'country': self._rng_country_code()}, ensure_ascii=True),
                    json.dumps({'method': self._rng.choice(['card', 'paypal', 'bank']), 'provider': 'stripe'}, ensure_ascii=True),
                ])
                for row in rows:
                    item_writer.writerow(row)

                if item_id > n_items:
                    break

    def _generate_reviews(self, n_reviews: int, n_products: int, n_customers: int) -> None:
        seen: set[tuple[int, int]] = set()
        file, writer = self._open_csv_output('review', [])
        with file:
            review_id = 1
            while review_id <= n_reviews:
                product_id = self._rng.randint(1, n_products)
                customer_id = self._rng.randint(1, n_customers)
                if (product_id, customer_id) in seen:
                    continue
                seen.add((product_id, customer_id))
                writer.writerow([
                    review_id,
                    product_id,
                    customer_id,
                    self._rng.randint(1, 5),
                    self._rng_text(3, 7),
                    self._rng_text(20, 60),
                    iso(self._rng_timestamp_since(2)),
                    self._rng.randint(0, 200),
                ])
                review_id += 1


def export():
    return EdbtDataGenerator()
