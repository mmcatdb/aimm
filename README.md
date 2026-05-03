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
