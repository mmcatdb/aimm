# Dynamic Code

Contains all schema specific code. Each schema is in `{driver_type}/{schema_name}` subdirectory and includes:
- `query_registry.py`: Query templates (as a `QueryRegistry`)
- `loader.py`: Data loader
The `common/{schema_name}` subdirectory includes:
- `data_generator.py`: Data generator (if needed)
- Plus some shared code for the schema (if needed)
