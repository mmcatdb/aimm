import csv
import os
from decimal import InvalidOperation
from datetime import datetime
from common.driver_provider import DriverProvider
from common.drivers import MongoDriver, Neo4jDriver, PostgresDriver
from datasets.tpch.mongo_dao import MongoDAO
from datasets.tpch.neo4j_dao import Neo4jDAO
from datasets.tpch.postgres_dao import PostgresDAO
from common.config import Config

def main():
    config = Config.load()
    dbs = DriverProvider.default(config)
    # TODO
    schema_mapping = {
        'region': 'postgres',
        'nation': 'postgres',
        'customer': 'postgres',
        'orders': 'postgres',
        'lineitem': 'postgres',
        'part': 'postgres',
        'supplier': 'postgres',
        'partsupp': 'postgres',
    }
    populator = Populator(config, dbs, schema_mapping)

    try:
        schemas = define_schemas()
        # Entities should be populated in this order due to foreign key dependencies. So, they should be deleted in a reverse order.
        populate_order = [ 'region', 'nation', 'part', 'supplier', 'partsupp', 'customer', 'orders', 'lineitem' ]

        populator.run(schemas, populate_order)
    finally:
        dbs.close()
        print('\nDisconnected from all databases.')

class Populator:
    def __init__(self, config, dbs: DriverProvider, schema_mapping: dict[str, str]):
        self._config = config
        self.schema_mapping = schema_mapping
        self.daos = {
            'postgres': PostgresDAO(dbs.get_typed('postgres', PostgresDriver)),
            'mongo': MongoDAO(dbs.get_typed('mongo', MongoDriver)),
            'neo4j': Neo4jDAO(dbs.get_typed('neo4j', Neo4jDriver)),
        }

    def run(self, schemas: dict[str, list[dict]], populate_order: list[str]):
        self.drop_all(populate_order)

        self.create_schemas(schemas)

        self.create_data(populate_order, schemas)

    def drop_all(self, populate_order: list[str]):
        by_schema = dict()
        for entity in populate_order:
            db_type = self.schema_mapping.get(entity)
            if db_type:
                if db_type not in by_schema:
                    by_schema[db_type] = []
                by_schema[db_type].append(entity)

        for db_type, entity_populate_order in by_schema.items():
            dao = self.daos[db_type]
            dao.drop_kinds(entity_populate_order)

    def create_schemas(self, schemas: dict[str, list[dict]]):
        for entity, schema in schemas.items():
            if entity in self.schema_mapping:
                dao = self.get_dao_for_entity(entity)
                dao.create_kind_schema(entity, schema)
            else:
                print(f'Skipping schema creation for {entity}; not in schema_mapping.')

    def create_data(self, populate_order: list[str], schemas: dict[str, list[dict]]):
        for entity in populate_order:
            if entity in self.schema_mapping:
                self.populate_from_tbl(entity, schemas[entity])
            else:
                print(f'Skipping {entity}; add to schema_mapping to populate.')

    def populate_from_tbl(self, entity: str, schema: list[dict]):
        filename = entity + '.tbl'
        path = os.path.join(self._config.import_directory, filename)

        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                # Skip empty or malformed rows
                if not row or all(col == '' for col in row):
                    continue

                data = {}
                for i, column in enumerate(schema):
                    if i < len(row) and column['name']:
                        value = row[i]
                        # Apply type conversion for MongoDB
                        db_type = self.schema_mapping[entity]
                        if db_type == 'mongodb':
                            value = convert_value(value, column['type'])

                        data[column['name']] = value

                # Drop potential empty key from final delimiter
                data = {k: v for k, v in data.items() if k}
                if data:
                    dao = self.get_dao_for_entity(entity)
                    dao.insert(entity, data)

        print(f'Finished populating "{entity}" from "{path}"')

    def get_dao_for_entity(self, entity):
        db_type = self.schema_mapping[entity]
        assert db_type is not None, f'Entity "{entity}" not found in schema mapping. Add it to config.yaml under schema_mapping.'

        return self.daos[db_type]

def convert_value(value, data_type: str):
    """Convert a string value to the appropriate Python type based on schema type."""
    if value is None or value == '':
        return None

    data_type_upper = data_type.upper()

    try:
        if data_type_upper == 'INTEGER':
            return int(value)
        elif data_type_upper.startswith('DECIMAL'):
            return float(value)
        elif data_type_upper == 'DATE':
            # Parse date string (format: YYYY-MM-DD)
            return datetime.strptime(value, '%Y-%m-%d')
        elif data_type_upper.startswith('CHAR') or data_type_upper.startswith('VARCHAR'):
            return str(value).strip()
        else:
            return value
    except (ValueError, InvalidOperation) as e:
        print(f'Warning: Could not convert value "{value}" to type "{data_type}": {e}')
        return value

def define_schemas():
    region = [
        {'name': 'r_regionkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'r_name', 'type': 'CHAR(25)'},
        {'name': 'r_comment', 'type': 'VARCHAR(152)'}
    ]

    nation = [
        {'name': 'n_nationkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'n_name', 'type': 'CHAR(25)'},
        {'name': 'n_regionkey', 'type': 'INTEGER'},
        {'name': 'n_comment', 'type': 'VARCHAR(152)'}
    ]

    customer = [
        {'name': 'c_custkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'c_name', 'type': 'VARCHAR(25)'},
        {'name': 'c_address', 'type': 'VARCHAR(40)'},
        {'name': 'c_nationkey', 'type': 'INTEGER'},
        {'name': 'c_phone', 'type': 'CHAR(15)'},
        {'name': 'c_acctbal', 'type': 'DECIMAL(15,2)'},
        {'name': 'c_mktsegment', 'type': 'CHAR(10)'},
        {'name': 'c_comment', 'type': 'VARCHAR(117)'}
    ]

    orders = [
        {'name': 'o_orderkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'o_custkey', 'type': 'INTEGER'},
        {'name': 'o_orderstatus', 'type': 'CHAR(1)'},
        {'name': 'o_totalprice', 'type': 'DECIMAL(15,2)'},
        {'name': 'o_orderdate', 'type': 'DATE'},
        {'name': 'o_orderpriority', 'type': 'CHAR(15)'},
        {'name': 'o_clerk', 'type': 'CHAR(15)'},
        {'name': 'o_shippriority', 'type': 'INTEGER'},
        {'name': 'o_comment', 'type': 'VARCHAR(79)'}
    ]

    lineitem = [
        {'name': 'l_orderkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'l_partkey', 'type': 'INTEGER'},
        {'name': 'l_suppkey', 'type': 'INTEGER'},
        {'name': 'l_linenumber', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'l_quantity', 'type': 'DECIMAL(15,2)'},
        {'name': 'l_extendedprice', 'type': 'DECIMAL(15,2)'},
        {'name': 'l_discount', 'type': 'DECIMAL(15,2)'},
        {'name': 'l_tax', 'type': 'DECIMAL(15,2)'},
        {'name': 'l_returnflag', 'type': 'CHAR(1)'},
        {'name': 'l_linestatus', 'type': 'CHAR(1)'},
        {'name': 'l_shipdate', 'type': 'DATE'},
        {'name': 'l_commitdate', 'type': 'DATE'},
        {'name': 'l_receiptdate', 'type': 'DATE'},
        {'name': 'l_shipinstruct', 'type': 'CHAR(25)'},
        {'name': 'l_shipmode', 'type': 'CHAR(10)'},
        {'name': 'l_comment', 'type': 'VARCHAR(44)'}
    ]

    part = [
        {'name': 'p_partkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'p_name', 'type': 'VARCHAR(55)'},
        {'name': 'p_mfgr', 'type': 'CHAR(25)'},
        {'name': 'p_brand', 'type': 'CHAR(10)'},
        {'name': 'p_type', 'type': 'VARCHAR(25)'},
        {'name': 'p_size', 'type': 'INTEGER'},
        {'name': 'p_container', 'type': 'CHAR(10)'},
        {'name': 'p_retailprice', 'type': 'DECIMAL(15,2)'},
        {'name': 'p_comment', 'type': 'VARCHAR(23)'}
    ]

    supplier = [
        {'name': 's_suppkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 's_name', 'type': 'CHAR(25)'},
        {'name': 's_address', 'type': 'VARCHAR(40)'},
        {'name': 's_nationkey', 'type': 'INTEGER'},
        {'name': 's_phone', 'type': 'CHAR(15)'},
        {'name': 's_acctbal', 'type': 'DECIMAL(15,2)'},
        {'name': 's_comment', 'type': 'VARCHAR(101)'}
    ]

    partsupp = [
        {'name': 'ps_partkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'ps_suppkey', 'type': 'INTEGER', 'primary_key': True},
        {'name': 'ps_availqty', 'type': 'INTEGER'},
        {'name': 'ps_supplycost', 'type': 'DECIMAL(15,2)'},
        {'name': 'ps_comment', 'type': 'VARCHAR(255)'}
    ]

    return {
        'region': region,
        'nation': nation,
        'customer': customer,
        'orders': orders,
        'lineitem': lineitem,
        'part': part,
        'supplier': supplier,
        'partsupp': partsupp
    }

if __name__ == '__main__':
    main()
