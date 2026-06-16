#!/bin/bash
set -euo pipefail

python -m scripts.flat.mongo.create_dataset mongo/tpch-2-flat tpch-2/measured-1000-30.jsonl

python -m scripts.flat.mongo.create_dataset \
  mongo/tpch-2-flat-train \
  tpch-2/measured-1000-5.jsonl \
  --val-dataset mongo/tpch-2-flat-val \
  --val-ratio 0.2 \
  --split-seed 42 \
  --skip-first 15

python -m scripts.flat.mongo.create_dataset \
  mongo/edbt-3-flat \
  edbt-3/measured-1000-40.jsonl \
  --feature-extractor-dataset tpch-2-flat-train \
  --refresh-queryplanner \
  --skip-first 15

python -m scripts.flat.mongo.train \
  mongo/tpch-2-flat-xgb-log \
  tpch-2-flat-train \
  tpch-2-flat-val \
  --model-type random_forest 
  # --n-estimators 800 \
  # --max-depth 6 \
  # --learning-rate 0.03 \
  # --sample-weight log_latency


python -m scripts.flat.mongo.test \
  mongo/tpch-2-flat-xgb-log \
  edbt-3-flat
