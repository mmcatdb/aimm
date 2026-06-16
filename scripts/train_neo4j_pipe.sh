#!/bin/bash
set -euo pipefail

python -m scripts.flat.neo4j.create_dataset \
  neo4j/tpch-2-flat-train \
  tpch-2/measured-1000-40.jsonl \
  --val-dataset neo4j/tpch-2-flat-val \
  --val-ratio 0.2 \
  --skip-first 20


python -m scripts.flat.neo4j.create_dataset \
  neo4j/edbt-3-flat \
  edbt-3/measured-1000-40.jsonl \
  --feature-extractor-dataset tpch-2-flat-train \
  --skip-first 20

python -m scripts.flat.neo4j.train \
  neo4j/tpch-2-flat-rf \
  tpch-2-flat-train \
  tpch-2-flat-val \
  --model-type random_forest

python -m scripts.flat.neo4j.test \
  neo4j/tpch-2-flat-rf \
  edbt-1-2-3-flat-art-fe-test
