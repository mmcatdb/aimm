#!/bin/bash
set -euo pipefail

python -m scripts.flat.neo4j.create_dataset \
  neo4j/art-1-2-3-flat-train \
  art-1/measured-1000-15.jsonl \
  art-2/measured-1000-15.jsonl \
  art-3/measured-1000-15.jsonl \
  --skip-first 10

python -m scripts.flat.neo4j.create_dataset \
  neo4j/tpch-2-flat-art-fe-val \
  tpch-2/measured-1000-40.jsonl \
  --feature-extractor-dataset art-1-2-3-flat-train \
  --skip-first 25

python -m scripts.flat.neo4j.create_dataset \
  neo4j/edbt-1-2-3-flat-art-fe-test \
  edbt-1/measured-1000-40.jsonl \
  edbt-2/measured-1000-40.jsonl \
  edbt-3/measured-1000-40.jsonl \
  --feature-extractor-dataset art-1-2-3-flat-train \
  --skip-first 25

python -m scripts.flat.neo4j.train \
  neo4j/art-1-2-3-flat-rf \
  art-1-2-3-flat-train \
  tpch-2-flat-art-fe-val \
  --model-type random_forest \
  --max-depth 8 \
  --min-samples-leaf 3 \
  --calibration thresholded_log_isotonic

python -m scripts.flat.neo4j.test \
  neo4j/art-1-2-3-flat-rf \
  edbt-1-2-3-flat-art-fe-test
