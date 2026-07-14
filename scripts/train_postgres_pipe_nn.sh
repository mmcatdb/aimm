#!/bin/bash
set -euo pipefail

python -m scripts.pipeline.validate_edbt_measurements \
  postgres \
  edbt-3/measured-700-40.jsonl

python -m scripts.neural.create_dataset \
  postgres/art-1-4-train \
  art-1/measured-700-40.jsonl \
  art-2/measured-700-40.jsonl \
  art-3/measured-700-40.jsonl \
  --skip-first 25

python -m scripts.neural.create_dataset \
  postgres/tpch-2-art-fe-val \
  tpch-2/measured-700-40.jsonl \
  --feature-extractor-dataset art-1-4-train \
  --skip-first 25

python -m scripts.neural.create_dataset \
  postgres/edbt-1-4-art-fe-test \
  edbt-3/measured-700-40.jsonl \
  --feature-extractor-dataset art-1-4-train \
  --skip-first 25

python -m scripts.neural.train \
  postgres/art-1-4-model \
  art-1-4-train \
  tpch-2-art-fe-val

python -m scripts.neural.test \
  postgres/art-1-4-model/best \
  edbt-1-4-art-fe-test
