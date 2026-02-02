import json
import math
from typing_extensions import override
from common.config import Config
from dataclasses import dataclass
from datetime import timedelta
from common.data_generator import AliasSampler, DataGenerator, clamp_int, iso

class EdbtDataGenerator(DataGenerator):
    """
    Data generator for the EDBT dataset.
    - Not all kinds scale the same (some grow faster).
    - Top ~1% products are "hot" and get more orders.

    Kinds: users, sellers, products, categories, has_category, has_interest, follows, orders, order_items, reviews, similar.
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
        self._generate_users(c.users)
        self._generate_sellers(c.sellers)
        self._generate_categories(c.categories)

        # Products
        seller_by_product, price_by_product, active_by_product = self._generate_products(c.products, c.sellers)

        # Many-to-many / graphy tables
        self._generate_has_category(c.products, c.categories)
        self._generate_has_interest(c.users, c.categories)
        self._generate_follows(c.users, c.follows)

        # Skewed sampler for product picks in orders
        weights = self._build_product_weights(c.products)
        product_sampler = self._create_sampler(weights)

        # Orders and order items (stream)
        self._generate_orders_and_items(
            c.orders,
            c.users,
            seller_by_product,
            price_by_product,
            active_by_product,
            product_sampler,
        )

        # Reviews
        self._generate_reviews(c.reviews, c.users, c.products)

        # Similar pairs
        self._generate_similar(c.products, c.similar_pairs)

    def _generate_counts(self) -> 'Counts':
        """
        scale=1 aims for "several MB total" in csv.
        Bigger scale multiplies size.
        Not all kinds scale the same.
        """
        return Counts(
            users = self._scaled(50_000, 1.00),
            sellers = self._scaled(5_000, 0.90),
            products = self._scaled(20_000, 0.95),
            categories = self._scaled(2_000, 0.60),
            orders = self._scaled(200_000, 1.05),
            reviews = self._scaled(300_000, 1.00),
            follows = self._scaled(200_000, 1.10),
            similar_pairs = self._scaled(120_000, 0.90),
        )

    def _generate_users(self, n_users: int) -> None:
        f, w = self._open_csv('users', ['user_id', 'handle', 'email', 'created_at', 'country_code', 'is_active', 'profile'])

        countries = ['US', 'GB', 'DE', 'FR', 'CZ', 'PL', 'ES', 'IT', 'NL', 'SE', 'IN', 'BR', 'CA', 'AU']

        for user_id in range(1, n_users + 1):
            handle = f'user{user_id}'
            email = f'{handle}@example.com'
            created_at = self._rand_ts_since(3)
            country = countries[self._rng.randrange(len(countries))]
            is_active = True if self._rng.random() < 0.98 else False
            profile = {
                'bio': f'Bio of {handle}',
                'tz': 'UTC',
                'lang': 'en',
            }
            w.writerow([user_id, handle, email, iso(created_at), country, str(is_active).lower(), json.dumps(profile, ensure_ascii=False)])

        f.close()

    def _generate_sellers(self, n_sellers: int) -> None:
        f, w = self._open_csv('sellers', ['seller_id', 'display_name', 'created_at', 'country_code', 'is_active'])

        countries = ['US', 'GB', 'DE', 'FR', 'CZ', 'PL', 'ES', 'IT', 'NL', 'SE']

        for seller_id in range(1, n_sellers + 1):
            name = f'Seller {seller_id}'
            created_at = self._rand_ts_since(5)
            country = countries[self._rng.randrange(len(countries))]
            is_active = True if self._rng.random() < 0.97 else False
            w.writerow([seller_id, name, iso(created_at), country, str(is_active).lower()])

        f.close()

    def _generate_categories(self, n_categories: int) -> None:
        """
        Builds a simple tree.
        We store path as a slash path, like: /root12/sub3/sub1
        """
        f, w = self._open_csv('categories', ['category_id', 'parent_category_id', 'name', 'path'])

        # Create roots first
        n_roots = clamp_int(int(round(math.sqrt(n_categories))), 10, 200)
        roots = []
        for i in range(1, n_roots + 1):
            cid = i
            name = f'Category {cid}'
            path = f'/cat{cid}'
            roots.append((cid, path))
            w.writerow([cid, '', name, path])

        # Fill the rest by picking a parent from existing nodes
        paths = {cid: path for cid, path in roots}
        for cid in range(n_roots + 1, n_categories + 1):
            parent = self._rng.randrange(1, cid)  # parent among existing
            parent_path = paths[parent]
            name = f'Category {cid}'
            path = f'{parent_path}/cat{cid}'
            paths[cid] = path
            w.writerow([cid, parent, name, path])

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
        f, w = self._open_csv('products', ['product_id', 'seller_id', 'sku', 'title', 'description', 'price_cents', 'currency','stock_qty', 'is_active', 'created_at', 'updated_at', 'attributes']
        )

        seller_id_by_product = [0] * n_products
        price_by_product = [0] * n_products
        active_by_product = [True] * n_products

        for product_id in range(1, n_products + 1):
            seller_id = self._rng.randint(1, n_sellers)
            sku = f'SKU-{product_id:09d}'
            title = f'Product {product_id}'
            description = f'Description for product {product_id}'
            # Price: mostly low, some higher
            base = int(round(500 + (self._rng.random() ** 2) * 20_000))  # cents
            price_cents = clamp_int(base, 100, 200_000)
            currency = 'USD'
            # Stock: many small, some big
            stock_qty = int(round((self._rng.random() ** 1.7) * 200))
            # Make hot products have more stock so they can sell
            if product_id <= max(1, n_products // 100):
                stock_qty += 500 + self._rng.randint(0, 1500)

            is_active = True if self._rng.random() < 0.96 else False
            created_at = self._rand_ts_since(5)
            updated_at = created_at + timedelta(days=self._rng.randint(0, 60))

            attrs = {
                'brand': f'Brand {self._rng.randint(1, 200)}',
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
                w.writerow([product_id, cid, iso(self._rand_ts_since(1))])

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
                w.writerow([user_id, cid, strength, iso(self._rand_ts_since(2))])

        f.close()

    def _generate_follows(self, n_users: int, n_edges: int) -> None:
        """
        Directed edges. No self follows.
        We keep duplicates low by making edges per user with a local set.
        """
        f, w = self._open_csv('follows', ['follower_user_id', 'followee_user_id', 'created_at'])

        # Spread edges across users
        edges_written = 0
        user_id = 1
        while edges_written < n_edges and user_id <= n_users:
            # Each user follows a small number
            k = clamp_int(int((self._rng.random() ** 1.2) * 25), 0, 25)
            if edges_written + k > n_edges:
                k = n_edges - edges_written

            chosen = set()
            while len(chosen) < k:
                followee = self._rng.randint(1, n_users)
                if followee != user_id:
                    chosen.add(followee)

            for followee in chosen:
                w.writerow([user_id, followee, iso(self._rand_ts_since(3))])
                edges_written += 1
                if edges_written >= n_edges:
                    break

            user_id += 1

        f.close()

    def _generate_orders_and_items(self, n_orders: int, n_users: int, seller_by_product: list[int], price_by_product: list[int], active_by_product: list[bool], product_sampler: 'AliasSampler') -> None:
        """
        Stream orders and items together.
        That avoids huge memory for totals.
        """
        fo, wo = self._open_csv('orders', ['order_id', 'buyer_user_id', 'order_ts', 'status', 'total_cents', 'currency', 'shipping', 'payment'])
        fi, wi = self._open_csv('order_items', ['order_item_id', 'order_id', 'product_id', 'seller_id', 'unit_price_cents', 'quantity','created_at', 'product_snapshot']
        )

        statuses = ['paid', 'shipped', 'cancelled', 'refunded']
        status_w = [0.70, 0.20, 0.07, 0.03]

        order_item_id = 1

        for order_id in range(1, n_orders + 1):
            buyer = self._rng.randint(1, n_users)
            ts = self._rand_ts_since(1)
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

                snapshot = {'title': f'Product {product_id}', 'price_cents': unit_price, 'currency': 'USD'}

                wi.writerow([
                    order_item_id,
                    order_id,
                    product_id,
                    seller_id,
                    unit_price,
                    qty,
                    iso(ts),
                    json.dumps(snapshot, ensure_ascii=False),
                ])
                order_item_id += 1

            shipping = {'method': self._rng.choice(['standard', 'express']), 'country': self._rng.choice(['US', 'GB', 'DE', 'CZ'])}
            payment = {'method': self._rng.choice(['card', 'paypal', 'bank']), 'provider': self._rng.choice(['stripe', 'adyen', 'braintree'])}

            wo.writerow([order_id, buyer, iso(ts), status, total, 'USD', json.dumps(shipping, ensure_ascii=False), json.dumps(payment, ensure_ascii=False)])

        fo.close()
        fi.close()

    def _generate_reviews(self, n_reviews: int, n_users: int, n_products: int) -> None:
        """
        Reviews are big.
        We do per-product review counts, then pick users for that product.
        This avoids duplicates for (product_id, user_id).
        """
        f, w = self._open_csv('reviews', ['review_id', 'product_id', 'user_id', 'rating', 'title', 'body', 'created_at', 'helpful_votes'])

        years = 2

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

            # Random around mean, but keep it small per product
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
                title = f'Review for product {product_id}'
                body = f'I rate product {product_id} with {rating} stars.'
                created_at = self._rand_ts_since(years)
                helpful = int(round((self._rng.random() ** 2.0) * 50))
                w.writerow([review_id, product_id, user_id, rating, title, body, iso(created_at), helpful])
                review_id += 1

            remaining -= len(chosen_users)

        # If we still have budget, add more to hot products
        product_id = 1
        while remaining > 0:
            pid = 1 + (product_id % max(1, hot_n))
            user_id = self._rng.randint(1, n_users)
            rating = clamp_int(int(round(1 + (self._rng.random() ** 0.6) * 4)), 1, 5)
            title = f'Review for product {pid}'
            body = f'I rate product {pid} with {rating} stars.'
            created_at = self._rand_ts_since(years)
            helpful = int(round((self._rng.random() ** 2.0) * 50))
            w.writerow([review_id, pid, user_id, rating, title, body, iso(created_at), helpful])
            review_id += 1
            remaining -= 1
            product_id += 1

        f.close()

    def _generate_similar(self, n_products: int, n_pairs: int) -> None:
        """
        Similar is symmetric and stored once (a < b).
        We generate a small number of neighbors per product for the first chunk,
        then fill until we reach n_pairs.
        """
        f, w = self._open_csv('similar', ['product_id_a', 'product_id_b', 'score', 'source', 'updated_at'])

        updated = self._now
        pairs_written = 0

        # First pass: local neighbors
        max_per_a = 8
        for a in range(1, n_products + 1):
            if pairs_written >= n_pairs:
                break
            k = clamp_int(int((self._rng.random() ** 0.6) * max_per_a), 0, max_per_a)

            chosen_b = set()
            tries = 0
            while len(chosen_b) < k and tries < k * 10:
                b = self._rng.randint(1, n_products)
                if b != a:
                    if a < b:
                        chosen_b.add(b)
                    else:
                        chosen_b.add(a)  # will be fixed below
                tries += 1

            for raw_b in chosen_b:
                b = raw_b
                if b == a:
                    continue
                a2, b2 = (a, b) if a < b else (b, a)
                if a2 == b2:
                    continue
                score = round(0.2 + (self._rng.random() ** 0.5) * 0.8, 6)
                w.writerow([a2, b2, score, 'model', iso(updated)])
                pairs_written += 1
                if pairs_written >= n_pairs:
                    break

        # Fill remaining with random pairs
        seen = set()
        while pairs_written < n_pairs:
            a = self._rng.randint(1, n_products)
            b = self._rng.randint(1, n_products)
            if a == b:
                continue
            a2, b2 = (a, b) if a < b else (b, a)
            key = (a2, b2)
            if key in seen:
                continue
            seen.add(key)
            score = round(0.2 + (self._rng.random() ** 0.5) * 0.8, 6)
            w.writerow([a2, b2, score, 'model', iso(updated)])
            pairs_written += 1

        f.close()

@dataclass
class Counts:
    users: int
    sellers: int
    products: int
    categories: int
    orders: int
    reviews: int
    follows: int
    similar_pairs: int
