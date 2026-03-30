import json
import math
from typing_extensions import override
from dataclasses import dataclass
from datetime import timedelta, datetime
from common.data_generator import AliasSampler, DataGenerator, clamp_int, iso

class EdbtDataGenerator(DataGenerator):
    """
    Data generator for the EDBT dataset.
    - Not all kinds scale the same (some grow faster).
    - Top ~1% products are "hot" and get more orders.

    Kinds: person, customer, seller, product, category, has_category, has_interest, follows, order, order_item, review.
    """
    @override
    def name(self):
        """Returns the name of the generator (for display purposes)."""
        return 'EDBT'

    @override
    def _generate_data(self):
        c = self.generate_counts()
        print('Counts:', c)

        # Base kinds
        persons = self._generate_persons(c.person)
        self._generate_sellers(c.seller)
        self._generate_categories(c.category)

        # Products
        products = self._generate_products(c.product, c.seller)

        # Many-to-many / graphy tables
        self._generate_has_category(c.product, c.category)
        self._generate_has_interest(c.person, c.category)
        self._generate_follows(c.person, c.follows)

        # Skewed sampler for product picks in orders
        weights = self._build_product_weights(c.product)
        product_sampler = self._create_sampler(weights)

        # Orders and order items (stream)
        customer_products = self._generate_customers_orders_items(
            c.order,
            persons,
            products,
            product_sampler,
        )

        # Reviews
        self._generate_reviews(c.review, customer_products)

    def generate_counts(self) -> 'Counts':
        """
        scale=1 aims for "several MB total" in csv.
        Bigger scale multiplies size.
        Not all kinds scale the same.
        """
        return Counts(
            person = self._scaled(50_000, 1.00),
            seller = self._scaled(5_000, 0.90),
            product = self._scaled(20_000, 0.95),
            category = self._scaled(2_000, 0.60),
            order = self._scaled(200_000, 1.05),
            review = self._scaled(100_000, 1.05),
            follows = self._scaled(200_000, 1.10),
        )

    def _generate_persons(self, n_persons: int) -> list[list]:
        """
        Returns arrays so customer snapshots can copy person data without rereading CSV.
        """
        f, w = self._open_csv_output('person', ['person_id', 'name', 'email', 'created_at', 'country_code', 'is_active', 'profile'])

        persons = list[list]()

        for person_id in range(1, n_persons + 1):
            name = self._rng_full_name()
            email = self._rng_unique_email(name)
            created_at = self._rng_timestamp_since(3)
            country = self._rng_country_code()
            is_active = True if self._rng.random() < 0.98 else False
            profile = {
                'bio': self._rng_text(5, 20),
                'tz': self._rng_time_zone(),
                'lang': self._rng_locale(),
            }
            person = [
                person_id,
                name,
                email,
                iso(created_at),
                country,
                str(is_active).lower(),
                json.dumps(profile, ensure_ascii=False)
            ]
            persons.append(person)
            w.writerow(person)

        f.close()

        return persons

    def _generate_sellers(self, n_sellers: int) -> None:
        f, w = self._open_csv_output('seller', ['seller_id', 'display_name', 'created_at', 'country_code', 'is_active'])

        for seller_id in range(1, n_sellers + 1):
            name = self._rng_full_name()
            created_at = self._rng_timestamp_since(5)
            country = self._rng_country_code()
            is_active = True if self._rng.random() < 0.97 else False
            w.writerow([seller_id, name, iso(created_at), country, str(is_active).lower()])

        f.close()

    def _generate_categories(self, n_categories: int) -> None:
        """
        Builds a simple tree.
        We store path as a slash path, like: /root12/sub3/sub1
        """
        f, w = self._open_csv_output('category', ['category_id', 'name', 'path'])

        # Create roots first
        n_roots = clamp_int(int(round(math.sqrt(n_categories))), 10, 200)
        roots = []
        for i in range(1, n_roots + 1):
            cid = i
            name = self._rng_word()
            path = f'/cat{cid}'
            roots.append((cid, path))
            w.writerow([cid, name, path])

        # Fill the rest by picking a parent from existing nodes
        paths = {cid: path for cid, path in roots}
        for cid in range(n_roots + 1, n_categories + 1):
            parent = self._rng.randrange(1, cid)  # parent among existing
            parent_path = paths[parent]
            name = self._rng_word()
            path = f'{parent_path}/cat{cid}'
            paths[cid] = path
            w.writerow([cid, name, path])

        f.close()

    def _build_product_weights(self, n_products: int) -> list[float]:
        """
        Heavy skew:
        - top 1% are hot
        - hot products get much higher pick prob
        """
        hot_n = max(1, n_products // 100)

        weights = [1.0] * n_products
        for i in range(hot_n):
            # Make hot items much heavier
            weights[i] = 40.0

        # Add gentle long-tail shape too (Zipf-ish)
        for i in range(n_products):
            rank = i + 1
            weights[i] *= 1.0 / (rank ** 0.20)

        return weights

    def _generate_products(self, n_products: int, n_sellers: int) -> list[list]:
        """
        Returns:
        - price_cents_by_product
        - is_active_by_product
        """
        f, w = self._open_csv_output('product', ['product_id', 'seller_id', 'sku', 'title', 'description', 'price_cents', 'currency', 'stock_qty', 'is_active', 'created_at', 'updated_at', 'attributes'])

        products = []

        for product_id in range(1, n_products + 1):
            seller_id = self._rng.randint(1, n_sellers)
            sku = f'SKU-{product_id:09d}'
            title = self._rng_text(2, 5)
            description = self._rng_text(10, 20)
            # Price: mostly low, some higher
            base = int(round(500 + (self._rng.random() ** 2) * 20_000))  # cents
            price_cents = clamp_int(base, 100, 200_000)
            currency = self._rng_currency()
            # Stock: many small, some big
            stock_qty = int(round((self._rng.random() ** 1.7) * 200))
            # Make hot products have more stock so they can sell
            if product_id <= max(1, n_products // 100):
                stock_qty += 500 + self._rng.randint(0, 1500)

            is_active = True if self._rng.random() < 0.96 else False
            created_at = self._rng_timestamp_since(5)
            updated_at = created_at + timedelta(days=self._rng.randint(0, 60))

            attrs = {
                'brand': self._rng_word(),
                'color': self._rng.choice(['black', 'white', 'red', 'blue', 'green']),
                'weight_g': int(50 + self._rng.random() * 2000),
            }

            product = [
                product_id,
                seller_id,
                sku,
                title,
                description,
                price_cents,
                currency,
                stock_qty,
                str(is_active).lower(),
                iso(created_at),
                iso(updated_at),
                json.dumps(attrs, ensure_ascii=False)
            ]
            w.writerow(product)
            products.append(product)

        f.close()

        return products

    def _generate_has_category(self, n_products: int, n_categories: int) -> None:
        f, w = self._open_csv_output('has_category', ['product_id', 'category_id', 'assigned_at'])

        for product_id in range(1, n_products + 1):
            # 1 to 3 categories per product
            k = 1 + (1 if self._rng.random() < 0.35 else 0) + (1 if self._rng.random() < 0.10 else 0)
            chosen = set()
            while len(chosen) < k:
                chosen.add(self._rng.randint(1, n_categories))
            for cid in chosen:
                w.writerow([product_id, cid, iso(self._rng_timestamp_since(1))])

        f.close()

    def _generate_has_interest(self, n_persons: int, n_categories: int) -> None:
        f, w = self._open_csv_output('has_interest', ['person_id', 'category_id', 'strength', 'created_at'])

        for person_id in range(1, n_persons + 1):
            # 1 to 6 interests
            k = clamp_int(int(1 + (self._rng.random() ** 0.4) * 6), 1, 6)
            chosen = set()
            while len(chosen) < k:
                chosen.add(self._rng.randint(1, n_categories))
            for cid in chosen:
                strength = clamp_int(int(1 + self._rng.random() * 10), 1, 10)
                w.writerow([person_id, cid, strength, iso(self._rng_timestamp_since(2))])

        f.close()

    def _generate_follows(self, n_persons: int, n_edges: int) -> None:
        """
        Directed edges. No self follows.
        We keep duplicates low by making edges per person with a local set.
        """
        f, w = self._open_csv_output('follows', ['from_id', 'to_id', 'created_at'])

        # Spread edges across persons
        edges_written = 0
        from_id = 1
        while edges_written < n_edges and from_id <= n_persons:
            # Each person follows a small number
            k = clamp_int(int((self._rng.random() ** 1.2) * 25), 0, 25)
            if edges_written + k > n_edges:
                k = n_edges - edges_written

            chosen = set()
            while len(chosen) < k:
                to_id = self._rng.randint(1, n_persons)
                if to_id != from_id:
                    chosen.add(to_id)

            for to_id in chosen:
                w.writerow([from_id, to_id, iso(self._rng_timestamp_since(3))])
                edges_written += 1
                if edges_written >= n_edges:
                    break

            from_id += 1

        f.close()

    def _generate_customers_orders_items(self, n_orders: int, persons: list[list], products: list[list], product_sampler: 'AliasSampler') -> list[tuple[datetime, list[int]]]:
        """
        Stream customers, orders and items together.
        That avoids huge memory for totals.
        Returns list of (customer_created_at, [product_ids]) for reviews later.
        """
        fc, wc = self._open_csv_output('customer', ['customer_id', 'person_id', 'snapshot_at', 'name', 'email', 'country_code', 'is_active', 'profile'])
        fo, wo = self._open_csv_output('order', ['order_id', 'customer_id', 'ordered_at', 'status', 'total_cents', 'currency', 'shipping', 'payment'])
        fi, wi = self._open_csv_output('order_item', ['order_item_id', 'order_id', 'product_id', 'unit_price_cents', 'quantity', 'line_total_cents', 'created_at'])

        statuses = ['paid', 'shipped', 'canceled', 'refunded']
        status_w = [0.70, 0.20, 0.07, 0.03]

        order_item_id = 1
        customer_id = 1
        customer_products = []

        for order_id in range(1, n_orders + 1):
            person = self._rng.choice(persons)
            ts = self._rng_timestamp_since(1)
            status = statuses[self._weighted_choice_int(status_w)]

            # Write customer snapshot (table per class child)
            wc.writerow([
                customer_id,
                person[0],
                iso(ts),
                person[1],
                person[2],
                person[4],
                person[5],
                person[6],
            ])

            # Items per order: mostly 1-3, sometimes bigger
            items_n = 1 + (1 if self._rng.random() < 0.30 else 0) + (1 if self._rng.random() < 0.12 else 0)
            if self._rng.random() < 0.03:
                items_n += self._rng.randint(2, 7)

            total = 0
            product_ids = []
            used_product_indexes = set()

            for _ in range(items_n):
                # Pick product with skew
                # Avoid duplicates inside one order, so it feels real
                for _tries in range(10):
                    p_idx = product_sampler.sample_index()
                    product = products[p_idx]
                    is_active = product[8] == 'true'
                    if p_idx not in used_product_indexes and is_active:
                        used_product_indexes.add(p_idx)
                        break
                else:
                    # fallback, even if dup
                    p_idx = product_sampler.sample_index()

                product = products[p_idx]

                qty = 1
                if self._rng.random() < 0.15:
                    qty = 2
                if self._rng.random() < 0.05:
                    qty = self._rng.randint(3, 6)

                unit_price = product[5]
                line = unit_price * qty
                total += line

                product_id = product[0]
                product_ids.append(product_id)

                # Write order item
                wi.writerow([
                    order_item_id,
                    order_id,
                    product_id,
                    unit_price,
                    qty,
                    line,
                    iso(ts),
                ])
                order_item_id += 1

            customer_products.append((ts, product_ids))

            shipping = {'method': self._rng.choice(['standard', 'express']), 'country': self._rng_country_code()}
            payment = {'method': self._rng.choice(['card', 'paypal', 'bank']), 'provider': self._rng.choice(['stripe', 'paypal', 'flowlance'])}

            # Write order
            wo.writerow([
                order_id,
                customer_id,
                iso(ts),
                status,
                total,
                self._rng_currency(),
                json.dumps(shipping, ensure_ascii=False),
                json.dumps(payment, ensure_ascii=False)
            ])
            customer_id += 1

        fc.close()
        fo.close()
        fi.close()

        return customer_products

    def _generate_reviews(self, n_reviews: int, customer_products: list[tuple[datetime, list[int]]]) -> None:
        """
        Reviews are big.
        We randomly select a customer and then one of his products. This should keep the distribution of reviews same as sold products.
        """
        f, w = self._open_csv_output('review', ['review_id', 'product_id', 'customer_id', 'rating', 'title', 'body', 'created_at', 'helpful_votes'])

        used_combinations = set[tuple[int, int]]()
        review_id = 1

        while review_id <= n_reviews:
            customer_index = self._rng.randrange(len(customer_products))
            customer = customer_products[customer_index]

            customer_id = customer_index + 1
            product_id = self._rng.choice(customer[1])

            if (customer_id, product_id) in used_combinations:
                continue
            used_combinations.add((customer_id, product_id))

            rating = clamp_int(int(round(1 + (self._rng.random() ** 0.6) * 4)), 1, 5)
            title = self._rng_text(4, 8)
            body = self._rng_text(20, 50)
            created_at = customer[0] + timedelta(days=self._rng.randint(0, 14))
            helpful = int(round((self._rng.random() ** 2.0) * 50))

            w.writerow([
                review_id,
                product_id,
                customer_id,
                rating,
                title,
                body,
                iso(created_at),
                helpful
            ])
            review_id += 1

        f.close()

@dataclass
class Counts:
    person: int
    seller: int
    product: int
    category: int
    order: int
    review: int
    follows: int
