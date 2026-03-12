#!/bin/bash

echo "Starting MongoDB..."
sudo systemctl start mongod

echo "Starting PostgreSQL..."
sudo systemctl start postgresql

echo "Starting Neo4j..."
sudo systemctl start neo4j