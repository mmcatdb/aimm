#!/bin/bash

dataset_name=$1
echo "Running pipeline for dataset: $dataset_name"

python -m scripts.create_dataset postgres/$dataset_name-train   $dataset_name/measured-1000-2.jsonl   --val-dataset postgres/$dataset_name-val   --val-ratio 0.2   --split-seed 69
# python -m scripts.create_dataset postgres/$dataset_name $dataset_name/measured-1000-2.jsonl

python -m scripts.train postgres/$dataset_name-model $dataset_name-train $dataset_name-val