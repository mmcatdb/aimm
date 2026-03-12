# aimm

## Configuration

- Create configuration file:
```bash
cp .env.example .env
```
- Fill in and adjust configuration in `.env` as needed (e.g., ports, passwords).
- Start databases using Docker Compose:
```bash
docker compose up -d
```

## Python stuff

- Create virtual environment (needs to be done only once):
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

## Input data

### EDBT (generated)

```bash
python -m scripts.generate_data edbt --scale 0.1
python -m scripts.populate_db postgres edbt
python -m scripts.populate_db neo4j edbt
```

### TPC-H (downloaded)

- Download [TPC-H data](https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/tpch-data-small.zip) and extract it into the `data/inputs/tpch` directory.
```bash
python -m scripts.populate_db postgres tpch
python -m scripts.populate_db neo4j tpch
```

## Experiments

```bash
python -m experiments check
python -m experiments evaluate -c data/checkpoints/tpch_neo4j_final.pt -d neo4j
```
