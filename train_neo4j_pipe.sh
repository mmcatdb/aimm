#!/bin/bash
set -euo pipefail

python -m scripts.create_neo4j_flat_dataset \
  neo4j/tpch-2-flat-train \
  tpch-2/measured-1000-5.jsonl \
  --val-dataset neo4j/tpch-2-flat-val \
  --val-ratio 0.2 \
  --split-seed 69

python -m scripts.create_neo4j_flat_dataset \
  neo4j/edbt-3-flat \
  edbt-3/measured-1000-10.jsonl \
  --feature-extractor-dataset tpch-2-flat-train

python -m scripts.train_neo4j_flat \
  neo4j/tpch-2-flat-rf \
  tpch-2-flat-train \
  tpch-2-flat-val \
  --model-type random_forest

python -m scripts.test_neo4j_flat \
  neo4j/tpch-2-flat-rf \
  edbt-3-flat
