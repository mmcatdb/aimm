#!/bin/bash

# Test data generation, database population, and query measurement for all combinations of schemas and drivers.
# Uses scale 0 for quick testing.

scale=0
schemas=(
    art
    edbt
    tpch
)
drivers=(
    postgres
    mongo
    neo4j
)

function execute_command() {
    echo -e "\n\033[1;36mExecuting:\033[0m \033[0;33m$1\033[0m"
    eval "python -m $1"
}

for schema in "${schemas[@]}"; do
    echo -e "\033[1;36mTesting:\033[0m \033[0;32m$schema\033[0m"
    execute_command "scripts.generate_data $schema-$scale"

    for driver in "${drivers[@]}"; do
        execute_command "scripts.populate_db $driver/$schema-$scale"
        execute_command "scripts.measure_queries $driver/$schema-$scale --num-queries 0 --num-runs 1 --no-cache"
    done

    echo ""
done
