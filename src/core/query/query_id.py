from core.config import GLOBAL_RNG_SEED
from core.drivers import DriverType
from core.utils import deterministic_hash

# This is just a string instead of an enum so that the schemas can be added dynamically.
SchemaName = str
"""Identifies a schema.

The value is a valid python module name.
Example: `tpch`, `edbt`.
"""

SchemaId = str
"""Identifies a scaled schema data.

Pattern: {schema_name}-{scale}. Scale is a number. Generally, the data size grows as 2 ** scale.
Example: `tpch-1`, `edbt-10`.
"""

def create_schema_id(schema_name: SchemaName, scale: float) -> SchemaId:
    return f'{schema_name}-{scale:g}'

def parse_schema_id(id: SchemaId) -> tuple[SchemaName, float]:
    """Parses a schema id into `schema_name`, `scale`."""
    try:
        schema_name, scale_factor_str = id.split('-')
        return schema_name, float(scale_factor_str)
    except Exception as e:
        raise IdError.schema(id) from e

def create_schema_seed(schema_name: SchemaName, scale: float) -> int:
    """Creates a random generator for the specified schema and scale."""
    # The scale is reflected in the random generator so that smaller scales aren't just subsets of larger ones.
    # For some reason, this constant gives us the best distribution of random values for both data and query parameters.
    return GLOBAL_RNG_SEED + deterministic_hash(schema_name) + int(scale * 80085)

DatabaseId = str
"""Identifies a specific database instance to which we can connect and run queries on.

Pattern: {driver_type}/{schema_id}.
Example: `mongo/tpch-1`, `postgres/edbt-10`."""

def create_database_id_1(driver_type: DriverType, schema_id: SchemaId) -> DatabaseId:
    return f'{driver_type.value}/{schema_id}'

def create_database_id_2(driver_type: DriverType, schema_name: SchemaName, scale: float) -> DatabaseId:
    return create_database_id_1(driver_type, create_schema_id(schema_name, scale))

def parse_database_id(id: DatabaseId) -> tuple[DriverType, SchemaName, float]:
    """Parses a database id into `driver_type`, `schema_name`, `scale`."""
    try:
        driver_type_str, schema_id = id.split('/')
        schema_name, scale = parse_schema_id(schema_id)
        return DriverType(driver_type_str), schema_name, scale
    except Exception as e:
        raise IdError.schema(id) from e

TemplateName = str
"""Identifies a query template (within a schema and driver).

Can't not contain characters `/` and `:`, because they are used as separators in other identifiers.
Example: `complex`, `join-5`.
"""

# The point of this definition is to have the path (except for the template_name) to be a valid python path, so that we can import all templates for a given driver/schema combination.
# Differences in database indexes should be covered by different templates. I.e., some templates should filter over indexed columns, some shouldn't.
TemplateId = str
"""Identifies a query template.

Pattern: {driver_type}/{schema_name}/{template_name}.
Example: `mongo/tpch/complex`, `postgres/edbt/join-5`.
"""

def create_template_id(driver_type: DriverType, schema_name: SchemaName, template_name: TemplateName) -> TemplateId:
    return f'{driver_type.value}/{schema_name}/{template_name}'

def parse_template_id(id: TemplateId) -> tuple[DriverType, SchemaName, TemplateName]:
    """Parses a template id into `driver_type`, `schema_name`, `template_name`."""
    try:
        driver_type_str, schema_name, template_name = id.split('/')
        return (
            DriverType(driver_type_str),
            schema_name,
            template_name,
        )
    except Exception as e:
        raise IdError.template(id) from e

# Now we need to distinguish between different scales, because they might produce different params.
# E.g., when randomly selecting an userId, we need to know the max userId, which depends on the scale.
QueryInstanceId = str
"""Identifies a query instance.

Pattern: {database_id}/{template_name}:{instance_index}. Instance index is an integer.
Example: `mongo/tpch-1/complex:0`, `postgres/edbt-10/join-5:42`.
"""

def create_query_instance_id(database_id: DatabaseId, template_name: TemplateName, instance_index: int) -> QueryInstanceId:
    return f'{database_id}/{template_name}:{instance_index}'

def parse_query_instance_id(id: QueryInstanceId) -> tuple[DatabaseId, TemplateName, int]:
    """Parses a query instance id into its components."""
    try:
        database_id, rest = id.rsplit('/', 1)
        parse_database_id(database_id) # Just to validate the database id part.
        template_name, index_str = rest.split(':')

        return (
            database_id,
            template_name,
            int(index_str),
        )
    except Exception as e:
        raise IdError.query_instance(id) from e

def parse_query_instance_driver_type(id: QueryInstanceId) -> DriverType:
    """Extracts the driver type from a query instance id."""
    try:
        driver_type_str, _ = id.split('/')
        return DriverType(driver_type_str)
    except Exception as e:
        raise IdError.query_instance(id) from e

class IdError(ValueError):
    def __init__(self, id: str, type: str):
        super().__init__(f'Invalid {type} id: "{id}"')
        self.id = id
        self.type = type

    @staticmethod
    def schema(id: str) -> 'IdError':
        return IdError(id, 'schema')

    @staticmethod
    def database(id: str) -> 'IdError':
        return IdError(id, 'database')

    @staticmethod
    def template(id: str) -> 'IdError':
        return IdError(id, 'template')

    @staticmethod
    def query_instance(id: str) -> 'IdError':
        return IdError(id, 'query instance')
