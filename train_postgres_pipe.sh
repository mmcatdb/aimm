#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

dataset_name=$1

echo "Running flat random-forest pipeline for dataset: $dataset_name"

# python -m scripts.create_postgres_flat_dataset postgres/$dataset_name $dataset_name/measured-1000-2.jsonl  # If I only wanted to create a single dataset without a train/val split

python -m scripts.create_postgres_flat_dataset \
  "postgres/$dataset_name-flat-train" \
  "$dataset_name/measured-1000-2.jsonl" \
  --val-dataset "postgres/$dataset_name-flat-val" \
  --val-ratio 0.2 \
  --split-seed 69

python -m scripts.train_postgres_flat \
  "postgres/$dataset_name-flat-rf" \
  "$dataset_name-flat-train" \
  "$dataset_name-flat-val" \
  --model-type random_forest
