import csv
from decimal import Decimal, InvalidOperation
from datetime import datetime

import yaml
from daos.mongo_dao import MongoDAO
from daos.neo4j_dao import Neo4jDAO
from daos.postgres_dao import PostgresDAO


def convert_value(value, data_type):
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
        print(f"Warning: Could not convert value '{value}' to type '{data_type}': {e}")
        return value


class Populator:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.schema_mapping = self.config['schema_mapping']
        self.daos = {
            'postgres': PostgresDAO(self.config['postgres']),
            'mongodb': MongoDAO(self.config['mongodb']),
            'neo4j': Neo4jDAO(self.config['neo4j']),
        }

    def get_dao_for_entity(self, entity_name):
        db_type = self.schema_mapping.get(entity_name)
        if not db_type:
            raise ValueError(f"Entity '{entity_name}' not found in schema mapping. Add it to config.yaml under schema_mapping.")
        return self.daos[db_type]

    def populate_from_tbl(self, entity_name, file_path, schema):
        dao = self.get_dao_for_entity(entity_name)
        db_type = self.schema_mapping.get(entity_name)
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                # Skip empty or malformed rows
                if not row or all(col == '' for col in row): continue

                data = {}
                for i, column in enumerate(schema):
                    if i < len(row) and column['name']:
                        value = row[i]
                        # Apply type conversion for MongoDB
                        if db_type == 'mongodb':
                            value = convert_value(value, column['type'])
                        data[column['name']] = value

                # Drop potential empty key from final delimiter
                data = {k: v for k, v in data.items() if k}
                if data:
                    dao.insert(entity_name, data)
        
        print(f"Finished populating '{entity_name}' from '{file_path}'")
        
        
    def delete_entity(self, entity_name):
        dao = self.get_dao_for_entity(entity_name)
        dao.delete_all_from(entity_name)
        dao.drop_entity(entity_name)
        

    def disconnect_all(self):
        for dao in self.daos.values():
            dao.disconnect()
            

def main():
    populator = Populator()

    try:
        region_schema = [
            {'name': 'r_regionkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'r_name', 'type': 'CHAR(25)'},
            {'name': 'r_comment', 'type': 'VARCHAR(152)'}
        ]

        nation_schema = [
            {'name': 'n_nationkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'n_name', 'type': 'CHAR(25)'},
            {'name': 'n_regionkey', 'type': 'INTEGER'},
            {'name': 'n_comment', 'type': 'VARCHAR(152)'}
        ]


        customer_schema = [
            {'name': 'c_custkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'c_name', 'type': 'VARCHAR(25)'},
            {'name': 'c_address', 'type': 'VARCHAR(40)'},
            {'name': 'c_nationkey', 'type': 'INTEGER'},
            {'name': 'c_phone', 'type': 'CHAR(15)'},
            {'name': 'c_acctbal', 'type': 'DECIMAL(15,2)'},
            {'name': 'c_mktsegment', 'type': 'CHAR(10)'},
            {'name': 'c_comment', 'type': 'VARCHAR(117)'}
        ]
        
        orders_schema = [
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

        lineitem_schema = [
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

        part_schema = [
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

        supplier_schema = [
            {'name': 's_suppkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 's_name', 'type': 'CHAR(25)'},
            {'name': 's_address', 'type': 'VARCHAR(40)'},
            {'name': 's_nationkey', 'type': 'INTEGER'},
            {'name': 's_phone', 'type': 'CHAR(15)'},
            {'name': 's_acctbal', 'type': 'DECIMAL(15,2)'},
            {'name': 's_comment', 'type': 'VARCHAR(101)'}
        ]

        partsupp_schema = [
            {'name': 'ps_partkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'ps_suppkey', 'type': 'INTEGER', 'primary_key': True},
            {'name': 'ps_availqty', 'type': 'INTEGER'},
            {'name': 'ps_supplycost', 'type': 'DECIMAL(15,2)'},
            {'name': 'ps_comment', 'type': 'VARCHAR(255)'}
        ]

        # Delete existing entities if present
        for entity in ['lineitem', 'orders', 'customer', 'partsupp', 'supplier', 'part', 'nation', 'region']:
            if entity in populator.schema_mapping:
                try:
                    populator.delete_entity(entity)
                except Exception as e:
                    print(f"Skip delete for {entity}: {e}")

        create_schemas(populator, {
            'region': region_schema,
            'nation': nation_schema,
            'customer': customer_schema,
            'orders': orders_schema,
            'lineitem': lineitem_schema,
            'part': part_schema,
            'supplier': supplier_schema,
            'partsupp': partsupp_schema
        })

        populate_plan = [
            ('region', 'data/region.tbl', region_schema),
            ('nation', 'data/nation.tbl', nation_schema),
            ('part', 'data/part.tbl', part_schema),
            ('supplier', 'data/supplier.tbl', supplier_schema),
            ('partsupp', 'data/partsupp.tbl', partsupp_schema),
            ('customer', 'data/customer.tbl', customer_schema),
            ('orders', 'data/orders.tbl', orders_schema),
            ('lineitem', 'data/lineitem.tbl', lineitem_schema)
        ]

        for entity, path, schema in populate_plan:
            if entity in populator.schema_mapping:
                populator.populate_from_tbl(entity, path, schema)
            else:
                print(f"Skipping {entity}; add to schema_mapping to populate.")

    finally:
        populator.disconnect_all()
        print("\nDisconnected from all databases.")

def create_schemas(populator, schemas_dict):
    for entity_name, schema in schemas_dict.items():
        if entity_name in populator.schema_mapping:
            dao = populator.get_dao_for_entity(entity_name)
            dao.create_schema(entity_name, schema)
        else:
            print(f"Skipping schema creation for {entity_name}; not in schema_mapping.")

if __name__ == "__main__":
    main()
