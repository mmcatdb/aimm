import json
import math
from typing_extensions import override
from common.config import Config
from dataclasses import dataclass
from datetime import timedelta
from common.data_generator import AliasSampler, DataGenerator, clamp_int, iso

# TODO add isa user / person / customer
# TODO now we add foreign keys in neo4j (and then we remove them for some kinds)
# - they are probably used in some queries (in TPC-H) but they shouldn't (and after that, we should remove them)

class EdbtDataGenerator(DataGenerator):
    """
    Data generator for the EDBT dataset.
    - Not all kinds scale the same (some grow faster).
    - Top ~1% products are "hot" and get more orders.

    Kinds: user, seller, product, category, has_category, has_interest, follows, order, order_item, review.
    """
    def __init__(self, config: Config):
        super().__init__(config)

    @override
    def name(self):
        return 'EDBT'

    @override
    def _generate_data(self):
        c = self._generate_counts()

        c = self._generate_counts()
        print('Counts:', c)

        # Base kinds
        self._generate_users(c.user)
        self._generate_sellers(c.seller)
        self._generate_categories(c.category)

        # Products
        seller_by_product, price_by_product, active_by_product = self._generate_products(c.product, c.seller)

        # Many-to-many / graphy tables
        self._generate_has_category(c.product, c.category)
        self._generate_has_interest(c.user, c.category)
        self._generate_follows(c.user, c.follows)

        # Skewed sampler for product picks in orders
        weights = self._build_product_weights(c.product)
        product_sampler = self._create_sampler(weights)

        # Orders and order items (stream)
        self._generate_orders_and_items(
            c.order,
            c.user,
            seller_by_product,
            price_by_product,
            active_by_product,
            product_sampler,
        )

        # Reviews
        self._generate_reviews(c.review, c.user, c.product)

    def _generate_counts(self) -> 'Counts':
        """
        scale=1 aims for "several MB total" in csv.
        Bigger scale multiplies size.
        Not all kinds scale the same.
        """
        return Counts(
            user = self._scaled(50_000, 1.00),
            seller = self._scaled(5_000, 0.90),
            product = self._scaled(20_000, 0.95),
            category = self._scaled(2_000, 0.60),
            order = self._scaled(200_000, 1.05),
            review = self._scaled(300_000, 1.00),
            follows = self._scaled(200_000, 1.10),
        )

    def _generate_users(self, n_users: int) -> None:
        f, w = self._open_csv('user', ['user_id', 'handle', 'email', 'created_at', 'country_code', 'is_active', 'profile'])

        for user_id in range(1, n_users + 1):
            handle = self._rng_full_name()
            email = self._rng_email(handle)
            created_at = self._rng_timestamp_since(3)
            country = self._rng_country_code()
            is_active = True if self._rng.random() < 0.98 else False
            profile = {
                'bio': self._rng_text(5, 20),
                'tz': self._rng_time_zone(),
                'lang': self._rng_locale(),
            }
            w.writerow([user_id, handle, email, iso(created_at), country, str(is_active).lower(), json.dumps(profile, ensure_ascii=False)])

        f.close()

    def _generate_sellers(self, n_sellers: int) -> None:
        f, w = self._open_csv('seller', ['seller_id', 'display_name', 'created_at', 'country_code', 'is_active'])

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
        f, w = self._open_csv('category', ['category_id', 'name', 'path'])

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

    def _generate_products(self, n_products: int, n_sellers: int) -> tuple[list[int], list[int], list[bool]]:
        """
        Returns:
        - seller_id_by_product (index product_id-1)
        - price_cents_by_product
        - is_active_by_product
        """
        f, w = self._open_csv('product', ['product_id', 'seller_id', 'sku', 'title', 'description', 'price_cents', 'currency','stock_qty', 'is_active', 'created_at', 'updated_at', 'attributes'])

        seller_id_by_product = [0] * n_products
        price_by_product = [0] * n_products
        active_by_product = [True] * n_products

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

            seller_id_by_product[product_id - 1] = seller_id
            price_by_product[product_id - 1] = price_cents
            active_by_product[product_id - 1] = is_active

            w.writerow([
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
            ])

        f.close()
        return seller_id_by_product, price_by_product, active_by_product

    def _generate_has_category(self, n_products: int, n_categories: int) -> None:
        f, w = self._open_csv('has_category', ['product_id', 'category_id', 'assigned_at'])

        for product_id in range(1, n_products + 1):
            # 1 to 3 categories per product
            k = 1 + (1 if self._rng.random() < 0.35 else 0) + (1 if self._rng.random() < 0.10 else 0)
            chosen = set()
            while len(chosen) < k:
                chosen.add(self._rng.randint(1, n_categories))
            for cid in chosen:
                w.writerow([product_id, cid, iso(self._rng_timestamp_since(1))])

        f.close()

    def _generate_has_interest(self, n_users: int, n_categories: int) -> None:
        f, w = self._open_csv('has_interest', ['user_id', 'category_id', 'strength', 'created_at'])

        for user_id in range(1, n_users + 1):
            # 1 to 6 interests
            k = clamp_int(int(1 + (self._rng.random() ** 0.4) * 6), 1, 6)
            chosen = set()
            while len(chosen) < k:
                chosen.add(self._rng.randint(1, n_categories))
            for cid in chosen:
                strength = clamp_int(int(1 + self._rng.random() * 10), 1, 10)
                w.writerow([user_id, cid, strength, iso(self._rng_timestamp_since(2))])

        f.close()

    def _generate_follows(self, n_users: int, n_edges: int) -> None:
        """
        Directed edges. No self follows.
        We keep duplicates low by making edges per user with a local set.
        """
        f, w = self._open_csv('follows', ['from_id', 'to_id', 'created_at'])

        # Spread edges across users
        edges_written = 0
        from_id = 1
        while edges_written < n_edges and from_id <= n_users:
            # Each user follows a small number
            k = clamp_int(int((self._rng.random() ** 1.2) * 25), 0, 25)
            if edges_written + k > n_edges:
                k = n_edges - edges_written

            chosen = set()
            while len(chosen) < k:
                to_id = self._rng.randint(1, n_users)
                if to_id != from_id:
                    chosen.add(to_id)

            for to_id in chosen:
                w.writerow([from_id, to_id, iso(self._rng_timestamp_since(3))])
                edges_written += 1
                if edges_written >= n_edges:
                    break

            from_id += 1

        f.close()

    def _generate_orders_and_items(self, n_orders: int, n_users: int, seller_by_product: list[int], price_by_product: list[int], active_by_product: list[bool], product_sampler: 'AliasSampler') -> None:
        """
        Stream orders and items together.
        That avoids huge memory for totals.
        """
        fo, wo = self._open_csv('order', ['order_id', 'buyer_user_id', 'order_ts', 'status', 'total_cents', 'currency', 'shipping', 'payment'])
        fi, wi = self._open_csv('order_item', ['order_item_id', 'order_id', 'product_id', 'seller_id', 'unit_price_cents', 'quantity', 'line_total_cents', 'created_at', 'product_snapshot']
        )

        statuses = ['paid', 'shipped', 'cancelled', 'refunded']
        status_w = [0.70, 0.20, 0.07, 0.03]

        order_item_id = 1

        for order_id in range(1, n_orders + 1):
            buyer = self._rng.randint(1, n_users)
            ts = self._rng_timestamp_since(1)
            status = statuses[self._weighted_choice_int(status_w)]

            # Items per order: mostly 1-3, sometimes bigger
            items_n = 1 + (1 if self._rng.random() < 0.30 else 0) + (1 if self._rng.random() < 0.12 else 0)
            if self._rng.random() < 0.03:
                items_n += self._rng.randint(2, 7)

            total = 0
            used_products = set()

            for _ in range(items_n):
                # Pick product with skew
                # Avoid duplicates inside one order, so it feels real
                for _tries in range(10):
                    p_idx = product_sampler.sample_index()
                    product_id = p_idx + 1
                    if product_id not in used_products and active_by_product[p_idx]:
                        used_products.add(product_id)
                        break
                else:
                    # fallback, even if dup
                    product_id = product_sampler.sample_index() + 1

                p_idx = product_id - 1
                seller_id = seller_by_product[p_idx]
                unit_price = price_by_product[p_idx]

                qty = 1
                if self._rng.random() < 0.15:
                    qty = 2
                if self._rng.random() < 0.05:
                    qty = self._rng.randint(3, 6)

                line = unit_price * qty
                total += line

                snapshot = {'title': self._rng_text(2, 5), 'price_cents': unit_price, 'currency': self._rng_currency()}

                wi.writerow([
                    order_item_id,
                    order_id,
                    product_id,
                    seller_id,
                    unit_price,
                    qty,
                    line,
                    iso(ts),
                    json.dumps(snapshot, ensure_ascii=False),
                ])
                order_item_id += 1

            shipping = {'method': self._rng.choice(['standard', 'express']), 'country': self._rng_country_code()}
            payment = {'method': self._rng.choice(['card', 'paypal', 'bank']), 'provider': self._rng.choice(['stripe', 'paypal', 'flowlance'])}

            wo.writerow([order_id, buyer, iso(ts), status, total, self._rng_currency(), json.dumps(shipping, ensure_ascii=False), json.dumps(payment, ensure_ascii=False)])

        fo.close()
        fi.close()

    def _generate_reviews(self, n_reviews: int, n_users: int, n_products: int) -> None:
        """
        Reviews are big.
        We do per-product review counts, then pick users for that product.
        This avoids duplicates for (product_id, user_id).
        """
        f, w = self._open_csv('review', ['review_id', 'product_id', 'user_id', 'rating', 'title', 'body', 'created_at', 'helpful_votes'])

        years = 2

        used_combinations = set[tuple[int, int]]()

        # Create a 'review budget' per product.
        # Hot products get more.
        hot_n = max(1, n_products // 100)
        base_per_product = n_reviews / n_products

        review_id = 1
        remaining = n_reviews

        for product_id in range(1, n_products + 1):
            if remaining <= 0:
                break

            # Mean-ish count
            mean = base_per_product
            if product_id <= hot_n:
                mean *= 6.0

            # Random around mean, but keep it small per product.
            k = int(round(mean * (0.3 + self._rng.random())))
            k = clamp_int(k, 0, 60 if product_id <= hot_n else 15)
            if k > remaining:
                k = remaining

            chosen_users = set()
            tries = 0
            while len(chosen_users) < k and tries < k * 10:
                chosen_users.add(self._rng.randint(1, n_users))
                tries += 1

            for user_id in chosen_users:
                rating = clamp_int(int(round(1 + (self._rng.random() ** 0.6) * 4)), 1, 5)
                title = self._rng_text(4, 8)
                body = self._rng_text(20, 50)
                created_at = self._rng_timestamp_since(years)
                helpful = int(round((self._rng.random() ** 2.0) * 50))
                w.writerow([review_id, product_id, user_id, rating, title, body, iso(created_at), helpful])
                review_id += 1

                # No need to check here, we built chosen_users carefully.
                used_combinations.add((product_id, user_id))

            remaining -= len(chosen_users)

        # If we still have budget, add more to hot products.
        max_iterations = n_reviews * 10
        iteration = 0
        max_product_id = hot_n

        product_id = 1
        while remaining > 0:
            iteration += 1
            if iteration == max_iterations:
                print('Warning: reached max iterations while generating review. Continue with other product.')
                max_product_id = n_products

            pid = 1 + (product_id % max(1, max_product_id))
            user_id = self._rng.randint(1, n_users)

            # However, here we have to check for duplicates.
            combination = (pid, user_id)
            if combination in used_combinations:
                product_id += 1
                continue
            used_combinations.add(combination)

            rating = clamp_int(int(round(1 + (self._rng.random() ** 0.6) * 4)), 1, 5)
            title = self._rng_text(4, 8)
            body = self._rng_text(20, 50)
            created_at = self._rng_timestamp_since(years)
            helpful = int(round((self._rng.random() ** 2.0) * 50))
            w.writerow([review_id, pid, user_id, rating, title, body, iso(created_at), helpful])
            review_id += 1
            remaining -= 1
            product_id += 1

        f.close()

@dataclass
class Counts:
    user: int
    seller: int
    product: int
    category: int
    order: int
    review: int
    follows: int
