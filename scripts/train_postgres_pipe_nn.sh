#!/bin/bash
set -euo pipefail

# python -m scripts.create_dataset \
#   postgres/edbt-2-3-train \
#   edbt-2/measured-1000-20.jsonl \
#   edbt-3/measured-1000-20.jsonl \
#   --val-dataset postgres/edbt-2-3-val \
#   --val-ratio 0.2 \
#   --split-seed 69 \
#   --skip-first 12

# python -m scripts.create_dataset \
#   postgres/tpch-2-edbt-fe-test \
#   tpch-2/measured-1000-20.jsonl \
#   --feature-extractor-dataset edbt-2-3-train \
#   --skip-first 12

# python -m scripts.train \
#   postgres/edbt-2-3-model \
#   edbt-2-3-train \
#   edbt-2-3-val

# python -m scripts.test \
#   postgres/edbt-2-3-model/best \
#   tpch-2-edbt-fe-test



python -m scripts.create_dataset \
  postgres/edbt-2-3-train \
  tpch-2/measured-1000-20.jsonl \
  --val-dataset postgres/edbt-2-3-val \
  --val-ratio 0.2 \
  --split-seed 69 \
  --skip-first 12

python -m scripts.create_dataset \
  postgres/tpch-2-edbt-fe-test \
  edbt-2/measured-1000-20.jsonl \
  edbt-3/measured-1000-20.jsonl \
  --feature-extractor-dataset edbt-2-3-train \
  --skip-first 12

python -m scripts.train \
  postgres/edbt-2-3-model \
  edbt-2-3-train \
  edbt-2-3-val

python -m scripts.test \
  postgres/edbt-2-3-model/best \
  tpch-2-edbt-fe-test
