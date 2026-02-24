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
- Install dependencies (whenever they changes):
```bash
pip install -r requirements.txt
```
- Run scripts:
```bash
python -m path.to.file
```

## Input data

### EDBT (generated)

```bash
python -m datasets.edbt.data_generator --scale 1
```

### TPC-H (downloaded)

- Download [TPC-H data](https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/tpch-data-small.zip) and extract it into the `data/inputs` directory.
```bash
python -m database.populate_db
python -m database.populate_neo
```

## Experiments

```bash
python -m experiments check
python -m experiments evaluate
```
