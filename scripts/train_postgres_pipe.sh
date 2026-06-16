#!/bin/bash
set -euo pipefail

# python -m scripts.flat.postgres.create_dataset \
#   postgres/edbt-2-3-flat-train \
#   edbt-2/measured-1000-20.jsonl \
#   edbt-3/measured-1000-20.jsonl \
#   --val-dataset postgres/edbt-2-3-flat-val \
#   --val-ratio 0.2 \
#   --split-seed 69 \
#   --skip-first 12

# python -m scripts.flat.postgres.create_dataset \
#   postgres/tpch-2-flat-edbt-fe-test \
#   tpch-2/measured-1000-20.jsonl \
#   --feature-extractor-dataset edbt-2-3-flat-train \
#   --skip-first 12

# python -m scripts.flat.postgres.train \
#   postgres/edbt-2-3-flat-rf \
#   edbt-2-3-flat-train \
#   edbt-2-3-flat-val \
#   --model-type xgboost

# python -m scripts.flat.postgres.test \
#   postgres/edbt-2-3-flat-rf \
#   tpch-2-flat-edbt-fe-test


python -m scripts.flat.postgres.create_dataset \
  postgres/edbt-2-3-flat-train \
  tpch-2/measured-1000-20.jsonl \
  --val-dataset postgres/edbt-2-3-flat-val \
  --val-ratio 0.2 \
  --split-seed 69 \
  --skip-first 12

python -m scripts.flat.postgres.create_dataset \
  postgres/tpch-2-flat-edbt-fe-test \
  edbt-2/measured-1000-20.jsonl \
  edbt-3/measured-1000-20.jsonl \
  --feature-extractor-dataset edbt-2-3-flat-train \
  --skip-first 12

python -m scripts.flat.postgres.train \
  postgres/edbt-2-3-flat-rf \
  edbt-2-3-flat-train \
  edbt-2-3-flat-val \
  --model-type xgboost

python -m scripts.flat.postgres.test \
  postgres/edbt-2-3-flat-rf \
  tpch-2-flat-edbt-fe-test
