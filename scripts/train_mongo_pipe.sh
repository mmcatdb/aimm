#!/bin/bash
set -euo pipefail

python -m scripts.flat.mongo.create_dataset \
  mongo/art-1-2-flat-train \
  art-1/measured-2000-50.jsonl \
  art-2/measured-2000-50.jsonl \
  --refresh-queryplanner \
  --skip-first 30

python -m scripts.flat.mongo.create_dataset \
  mongo/tpch-2-flat-art-fe-val \
  tpch-2/measured-1000-50.jsonl \
  --feature-extractor-dataset art-1-2-flat-train \
  --refresh-queryplanner \
  --skip-first 30

python -m scripts.flat.mongo.create_dataset \
  mongo/edbt-3-flat-art-fe-test \
  edbt-3/measured-1000-50.jsonl \
  --feature-extractor-dataset art-1-2-flat-train \
  --refresh-queryplanner \
  --skip-first 30

python -m scripts.flat.mongo.train \
  mongo/art-1-2-flat-tail-blend \
  art-1-2-flat-train \
  tpch-2-flat-art-fe-val \
  --model-type tail_blend

python -m scripts.flat.mongo.test \
  mongo/art-1-2-flat-tail-blend \
  edbt-3-flat-art-fe-test
