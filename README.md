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

Generally, the following scripts work for all databases (`postgres`, `mongo`, `neo4j`). This guide uses `postgres` as an example, but feel free to replace it with the database of your choice.

### Data generation and population

Choose a *schema* (e.g., `art` and `edbt`) and a *scale* (a number >= 0). Use a small scale (e.g., 0) for testing. Then continue with larger scales for experiments. Generally, the data grows like 2^scale, so be careful.
- Generate data:
```bash
python -m scripts.pipeline.generate_data art-0
python -m scripts.pipeline.generate_data edbt-0
python -m scripts.pipeline.generate_data tpch-0
```
- Populate databases:
```bash
python -m scripts.pipeline.populate_db postgres/art-0
python -m scripts.pipeline.populate_db postgres/edbt-0
python -m scripts.pipeline.populate_db postgres/tpch-0
```

### Query measurement and plan extraction

- Run the following to generate queries, measure their latencies and extract their plans:
```bash
python -m scripts.pipeline.measure_queries postgres/art-0 --num-queries 1000 --num-runs 10
python -m scripts.pipeline.measure_queries postgres/edbt-0 --num-queries 100 --num-runs 10
```
- Adjust the number of queries and runs as needed.

### Dataset preparation

- Choose a cool dataset name (e.g., `train_dataset`) and run the following:
```bash
python -m scripts.neural.create_dataset postgres/train_dataset art-0/measured-1000-10.jsonl
```
- Multiple schema-scale combinations should be combined into a single dataset. So, if you also generated data for `art-2` with `200` queries and `20` runs, you should add `art-2/measured-200-20.jsonl` to the command above.
- A validation dataset (e.g., `val_dataset`) should reuse the feature extractor from the train dataset. Use:
```bash
python -m scripts.neural.create_dataset postgres/val_dataset --feature-extractor-dataset train_dataset edbt-1/measured-100-10.jsonl
```

### Training and evaluation

- Choose a fitting model name (e.g., `fitting_model`). Then, run the following:
```bash
python -m scripts.neural.train postgres/fitting_model train_dataset val_dataset
```
- To eavaluate a model, choose a checkpoint (e.g., `best`, `epoch/10`, ...) and a test dataset (e.g., `val_dataset`) and run:
```bash
python -m scripts.neural.test postgres/fitting_model/best val_dataset
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
python -m scripts.flat.postgres.create_dataset postgres/tpch-2-flat-train tpch-2/measured-1000-2.jsonl --val-dataset postgres/tpch-2-flat-val --val-ratio 0.2 --split-seed 69
```
- Train a random forest:
```bash
python -m scripts.flat.postgres.train postgres/tpch-2-flat-rf tpch-2-flat-train tpch-2-flat-val --model-type random_forest
```
- Or train an XGBoost regressor:
```bash
python -m scripts.flat.postgres.train postgres/tpch-2-flat-xgb tpch-2-flat-train tpch-2-flat-val --model-type xgboost
```
- Evaluate on a flat dataset:
```bash
python -m scripts.flat.postgres.test postgres/tpch-2-flat-rf edbt-2-flat
```
- Predict latency for a new query using plain `EXPLAIN` only:
```bash
python -m scripts.flat.postgres.predict postgres/tpch-2-flat-rf postgres/tpch-2 "SELECT * FROM orders LIMIT 10"
```

### MongoDB flat-feature tree models

MongoDB flat models use only `global_stats` and `queryPlanner` plans for
prediction. `executionStats` and Python query timing are used only by
`scripts.pipeline.measure_queries` when collecting training labels.

```bash
# Create train/validation flat datasets from measured MongoDB queries.
python -m scripts.flat.mongo.create_dataset mongo/tpch-2-flat-train tpch-2/measured-1000-5.jsonl --val-dataset mongo/tpch-2-flat-val --val-ratio 0.2 --split-seed 42 --refresh-queryplanner

# Create an EDBT test set with the training feature vocabulary.
python -m scripts.flat.mongo.create_dataset mongo/edbt-3-flat edbt-3/measured-1000-5.jsonl --feature-extractor-dataset tpch-2-flat-train --refresh-queryplanner

# Train and evaluate tree models.
python -m scripts.flat.mongo.train mongo/tpch-2-flat-rf tpch-2-flat-train tpch-2-flat-val --model-type random_forest
python -m scripts.flat.mongo.test mongo/tpch-2-flat-rf edbt-3-flat

# Predict a single query without executing it.
python -m scripts.flat.mongo.predict mongo/tpch-2-flat-rf mongo/tpch-2 '{"find":"orders","filter":{"o_orderkey":1}}'
```

### Neo4j flat-feature tree models

Neo4j flat models use only plain `EXPLAIN` plan fields for prediction:
estimated rows, operator structure, planner/runtime metadata, identifiers, and
operator `Details` patterns. `PROFILE` and timed query execution are allowed only
while collecting training labels; runtime counters such as rows, db hits,
page-cache hits, and operator time are not used as flat-model features.

```bash
# Create train/validation flat datasets from measured Neo4j queries.
python -m scripts.flat.neo4j.create_dataset neo4j/tpch-2-flat-train tpch-2/measured-1000-5.jsonl --val-dataset neo4j/tpch-2-flat-val --val-ratio 0.2 --split-seed 69

# Create an EDBT test set with the training feature vocabulary.
python -m scripts.flat.neo4j.create_dataset neo4j/edbt-3-flat edbt-3/measured-1000-10.jsonl --feature-extractor-dataset tpch-2-flat-train

# Train and evaluate tree models.
python -m scripts.flat.neo4j.train neo4j/tpch-2-flat-rf tpch-2-flat-train tpch-2-flat-val --model-type random_forest
python -m scripts.flat.neo4j.test neo4j/tpch-2-flat-rf edbt-3-flat

# Predict a single query without executing it.
python -m scripts.flat.neo4j.predict neo4j/tpch-2-flat-rf neo4j/tpch-2 "MATCH (n) RETURN n LIMIT 10"
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
