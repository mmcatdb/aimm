# aimm

## Configuration

- Create a configuration file:
```bash
cp .env.example .env
```
- Fill in and adjust configuration in `.env` as needed (e.g., ports, passwords).
- Start databases using Docker Compose:
```bash
docker compose up -d
```

## Python stuff

- Create virtual environment (should be done only once):
```bash
python -m venv .venv
```
- Activate virtual environment (needs to be done every new terminal session - python sucks):
```bash
source .venv/bin/activate
```
- Install dependencies (whenever they changes). Also needed for editable install:
```bash
pip install -e .
```
- Run scripts:
```bash
python -m path.to.file
```

## Workflow

Training latency estimation models requires data. For that, there is a whole pipeline of scripts. The general idea is to reuse the former steps as much as possible and repeat them only when necessary (their correspondent code or the previous steps have changed).

### Data generation and population

Choose a *schema* (e.g., `edbt`) and a *scale* (a number >= 0). Use a small scale (e.g., 0) for testing. Then continue with larger scales for experiments. Generally, the data grows like 2^scale, so be careful.
- Generate data:
```bash
python -m scripts.generate_data edbt-0
```
- Populate databases:
```bash
python -m scripts.populate_db postgres/edbt-0
python -m scripts.populate_db mongo/edbt-0
python -m scripts.populate_db neo4j/edbt-0
```
- Some schemas (e.g., TPC-H) are not generated but downloaded (at least partially). In that case, create manually the corresponding directories in `data/inputs` (see the `edbt-0` example) and put the downloaded data there. Then, run the generate script (if needed to generate the missing data) and populate the databases as usual.
- [Link](https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/tpch-data-small.zip) to some `tpch` data.

### Query measurement and plan extraction

- Run the following to generate queries, measure their latencies and extract their plans:
```bash
python -m scripts.measure_queries postgres/edbt-0 --num-queries 100 --num-runs 10
```
- Repeat for other databases (`mongo`, `neo4j`).
- Adjust the number of queries and runs as needed.

### Dataset preparation

- Choose a cool dataset name (e.g., `cool_dataset`) and run the following:
```bash
python -m scripts.create_dataset postgres/cool_dataset edbt-0/measured-100-10.jsonl
```
- Multiple schema-scale combinations should be combined into a single dataset. So, if you also generated data for `edbt-2` with `200` queries and `20` runs, you should add `edbt-2/measured-200-20.jsonl` to the command above.
- A validation dataset should reuse the feature extractor from the train dataset. Use:
```bash
python -m scripts.create_dataset postgres/hot_dataset --feature-extractor-dataset cool_dataset edbt-1/measured-100-10.jsonl
```

### Training and evaluation

- Choose a fitting model name (e.g., `fitting_model`). Then, run the following:
```bash
python -m scripts.train postgres/fitting_model cool_dataset hot_dataset
```
- To eavaluate a model, choose a checkpoint (e.g., `best`, `epoch/10`, ...) and a test dataset (e.g., `hot_dataset`) and run:
```bash
python -m scripts.test postgres/fitting_model/best hot_dataset
```

### PostgreSQL flat-feature tree models

PostgreSQL also has an alternative latency-estimation path based on fixed-length
features from `EXPLAIN` plans and scikit-learn style tree regressors. This path
coexists with the QPP-Net workflow above and does not require `EXPLAIN ANALYZE`
at inference time. Table and index names are excluded by default so models can be
evaluated across schemas more easily; pass `--include-schema-identifiers` when
you intentionally want schema-specific relation/index features.

- Create flat train/validation datasets from measured PostgreSQL queries:
```bash
python -m scripts.create_postgres_flat_dataset postgres/tpch-2-flat-train tpch-2/measured-1000-2.jsonl --val-dataset postgres/tpch-2-flat-val --val-ratio 0.2 --split-seed 69
```
- Train a random forest:
```bash
python -m scripts.train_postgres_flat postgres/tpch-2-flat-rf tpch-2-flat-train tpch-2-flat-val --model-type random_forest
```
- Or train an XGBoost regressor:
```bash
python -m scripts.train_postgres_flat postgres/tpch-2-flat-xgb tpch-2-flat-train tpch-2-flat-val --model-type xgboost
```
- Evaluate on a flat dataset:
```bash
python -m scripts.test_postgres_flat postgres/tpch-2-flat-rf edbt-2-flat-
```
- Predict latency for a new query using plain `EXPLAIN` only:
```bash
python -m scripts.predict_postgres_flat postgres/tpch-2-flat-rf postgres/tpch-2 "SELECT * FROM orders LIMIT 10"
```

## Experiments

- Explain query plan:
```bash
python -m scripts.show_plan postgres/tpch-1 "UPDATE orders SET o_totalprice = 0 WHERE o_orderkey = 1"
python -m scripts.show_plan postgres/tpch-1 basic-0
```

```bash
python -m experiments check
python -m experiments test postgres -c data/checkpoints/tpch_postgres_best.pt
```

## Development

- Use this to find all available operators and then find the missing ones:
```bash
python -m latency_estimation.neo4j train --dry-run
python -m scripts.show_plan neo4j edbt --all-queries
```
