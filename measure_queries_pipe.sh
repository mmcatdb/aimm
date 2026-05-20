#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <database> <dataset_name>"
  echo "Database must be one of: postgres, mongo, (neo4j coming soon hopefully)"
  exit 1
fi

database=$1
dataset_name=$2
num_queries=${3:-1000} # Optional argument for number of queries, default is 1000 if not provided
num_runs=${4:-2}       # Optional argument for number of runs, default is 2 if not provided

python -m scripts.generate_data "$dataset_name"
python -m scripts.populate_db "$database/$dataset_name"
python -m scripts.measure_queries "$database/$dataset_name" --num-queries $num_queries --num-runs $num_runs
