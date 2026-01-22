import argparse
import os
import shutil
import yaml
from neo4j import GraphDatabase

class TpchLoader:
    """
    A class to load TPC-H data into a Neo4j database.
    """
    def __init__(self, uri, user, password):
        """
        Initializes the loader and connects to the Neo4j database.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """
        Closes the database connection.
        """
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def run_query(self, query, database="neo4j", parameters=None):
        """
        Executes a Cypher query that doesn't need to return data.
        Ensures the result is fully consumed within the session.
        """
        with self.driver.session(database=database) as session:
            result = session.run(query, parameters or {})
            result.consume()

    def run_scalar(self, query, database="neo4j", parameters=None, key=None):
        """
        Executes a Cypher query expected to return a single record.
        Returns the value for 'key' or the first value in the record.
        """
        with self.driver.session(database=database) as session:
            rec = session.run(query, parameters or {}).single()
            if rec is None:
                return None
            if key is None:
                values = list(rec.values())
                return values[0] if values else None
            return rec.get(key)

    def reset_database(self):
        """
        Drops all constraints and deletes all nodes and relationships in batches.
        """
        print("Resetting database...")

        def get_constraint_names():
            query = "SHOW CONSTRAINTS YIELD name"
            with self.driver.session() as session:
                result = session.run(query)
                return [record["name"] for record in result]

        existing_constraints = get_constraint_names()
        constraints = [f"DROP CONSTRAINT {name} IF EXISTS" for name in existing_constraints]
        for constraint in constraints:
            try:
                self.run_query(constraint)
            except Exception as e:
                print(f"Constraint not found or could not be dropped: {constraint}... Error: {e}")

        print("Deleting all nodes and relationships in batches...")
        batch_size = 10_000
        total_deleted = 0

        delete_batch_query = """
        MATCH (n)
        WITH n LIMIT $limit
        WITH collect(n) AS nodes
        WITH nodes, size(nodes) AS deleted
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted
        """

        while True:
            deleted = self.run_scalar(delete_batch_query, parameters={"limit": batch_size}, key="deleted") or 0
            total_deleted += deleted
            print(f"Deleted batch: {deleted} nodes; total deleted so far: {total_deleted}")
            if deleted == 0:
                break

        # Verify emptiness and print out counts
        remaining_nodes = self.run_scalar("MATCH (n) RETURN count(n) AS nodes", key="nodes") or 0
        remaining_rels = self.run_scalar("MATCH ()-[r]-() RETURN count(r) AS rels", key="rels") or 0

        if remaining_nodes == 0 and remaining_rels == 0:
            print(f"Database has been cleared. Nodes: {remaining_nodes}, Relationships: {remaining_rels}")
        else:
            print(f"Warning: Database not empty after reset. Nodes: {remaining_nodes}, Relationships: {remaining_rels}")

    def create_constraints(self):
        """
        Creates unique constraints for primary keys to speed up data loading.
        """
        print("Creating constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.r_regionkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Nation) REQUIRE n.n_nationkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.c_custkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Order) REQUIRE o.o_orderkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.p_partkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.s_suppkey IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ps:PartSupp) REQUIRE (ps.p_partkey, ps.s_suppkey) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (li:LineItem) REQUIRE (li.o_orderkey, li.l_linenumber) IS UNIQUE"
        ]
        for query in queries:
            self.run_query(query)
        print("Constraints created.")

    def load_data(self):
        """
        Loads data from all .tbl files into Neo4j.
        The order of loading is important to ensure relationships can be formed.
        """
        print("\n--- Starting Data Loading ---")

        # Load nodes first
        self.load_regions()
        self.load_nations()
        self.load_parts()
        self.load_suppliers()
        self.load_customers()
        self.load_orders()

        # Load nodes and relationships for many-to-many tables
        self.load_partsupp()
        self.load_lineitems()
        
        # Create relationships for simple foreign keys
        self.create_nation_region_relationships()
        self.create_customer_nation_relationships()
        self.create_supplier_nation_relationships()
        self.create_order_customer_relationships()
        
        print("\n--- Data Loading Complete ---")
    
    # --- Individual Loading Functions ---

    def load_regions(self):
        print("Loading regions...")
        query = """
        LOAD CSV FROM 'file:///region.tbl' AS row FIELDTERMINATOR '|'
        CREATE (:Region {
            r_regionkey: toInteger(row[0]), 
            r_name: row[1], 
            r_comment: row[2]
        });
        """
        self.run_query(query)

    def load_nations(self):
        print("Loading nations...")
        query = """
        LOAD CSV FROM 'file:///nation.tbl' AS row FIELDTERMINATOR '|'
        CREATE (:Nation {
            n_nationkey: toInteger(row[0]), 
            n_name: row[1], 
            n_regionkey: toInteger(row[2]), 
            n_comment: row[3]
        });
        """
        self.run_query(query)

    def load_parts(self):
        print("Loading parts...")
        query = """
        LOAD CSV FROM 'file:///part.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            CREATE (:Part {
                p_partkey: toInteger(row[0]), 
                p_name: row[1], 
                p_mfgr: row[2], 
                p_brand: row[3], 
                p_type: row[4], 
                p_size: toInteger(row[5]), 
                p_container: row[6], 
                p_retailprice: toFloat(row[7]), 
                p_comment: row[8]
            })
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def load_suppliers(self):
        print("Loading suppliers...")
        query = """
        LOAD CSV FROM 'file:///supplier.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            CREATE (:Supplier {
                s_suppkey: toInteger(row[0]), 
                s_name: row[1], 
                s_address: row[2], 
                s_nationkey: toInteger(row[3]), 
                s_phone: row[4], 
                s_acctbal: toFloat(row[5]), 
                s_comment: row[6]
            })
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def load_customers(self):
        print("Loading customers...")
        query = """
        LOAD CSV FROM 'file:///customer.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            CREATE (:Customer {
                c_custkey: toInteger(row[0]), 
                c_name: row[1], 
                c_address: row[2], 
                c_nationkey: toInteger(row[3]), 
                c_phone: row[4], 
                c_acctbal: toFloat(row[5]), 
                c_mktsegment: row[6], 
                c_comment: row[7]
            })
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def load_orders(self):
        print("Loading orders...")
        query = """
        LOAD CSV FROM 'file:///orders.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            CREATE (:Order {
                o_orderkey: toInteger(row[0]), 
                o_custkey: toInteger(row[1]), 
                o_orderstatus: row[2], 
                o_totalprice: toFloat(row[3]), 
                o_orderdate: date(row[4]), 
                o_orderpriority: row[5], 
                o_clerk: row[6], 
                o_shippriority: toInteger(row[7]), 
                o_comment: row[8]
            })
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def load_partsupp(self):
        print("Loading partsupp and creating relationships to Part and Supplier...")
        query = """
        LOAD CSV FROM 'file:///partsupp.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            MATCH (p:Part {p_partkey: toInteger(row[0])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[1])})
            CREATE (p)<-[:IS_FOR_PART]-(ps:PartSupp {
                ps_availqty: toInteger(row[2]), 
                ps_supplycost: toFloat(row[3]), 
                ps_comment: row[4]
            })-[:SUPPLIED_BY]->(s)
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def load_lineitems(self):
        print("Loading lineitems and creating relationships to Order and PartSupp...")
        query = """
        LOAD CSV FROM 'file:///lineitem.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {
            MATCH (o:Order {o_orderkey: toInteger(row[0])})
            MATCH (p:Part {p_partkey: toInteger(row[1])})
            MATCH (s:Supplier {s_suppkey: toInteger(row[2])})
            MATCH (p)<-[:IS_FOR_PART]-(ps:PartSupp)-[:SUPPLIED_BY]->(s)
            CREATE (o)-[:CONTAINS_ITEM]->(li:LineItem {
                l_linenumber: toInteger(row[3]),
                l_quantity: toFloat(row[4]),
                l_extendedprice: toFloat(row[5]),
                l_discount: toFloat(row[6]),
                l_tax: toFloat(row[7]),
                l_returnflag: row[8],
                l_linestatus: row[9],
                l_shipdate: date(row[10]),
                l_commitdate: date(row[11]),
                l_receiptdate: date(row[12]),
                l_shipinstruct: row[13],
                l_shipmode: row[14],
                l_comment: row[15]
            })-[:IS_PRODUCT_SUPPLY]->(ps)
        } IN TRANSACTIONS OF 500 ROWS
        """
        self.run_query(query)

    def create_nation_region_relationships(self):
        print("Creating Nation -> Region relationships...")
        query = """
        MATCH (n:Nation), (r:Region {r_regionkey: n.n_regionkey})
        CREATE (n)-[:IS_IN_REGION]->(r);
        """
        self.run_query(query)
        # Remove redundant foreign key property
        self.run_query("MATCH (n:Nation) REMOVE n.n_regionkey;")
    
    def create_customer_nation_relationships(self):
        print("Creating Customer -> Nation relationships...")
        query = """
        MATCH (c:Customer), (n:Nation {n_nationkey: c.c_nationkey})
        CREATE (c)-[:IS_IN_NATION]->(n);
        """
        self.run_query(query)
        self.run_query("MATCH (c:Customer) REMOVE c.c_nationkey;")
        
    def create_supplier_nation_relationships(self):
        print("Creating Supplier -> Nation relationships...")
        query = """
        MATCH (s:Supplier), (n:Nation {n_nationkey: s.s_nationkey})
        CREATE (s)-[:IS_IN_NATION]->(n);
        """
        self.run_query(query)
        self.run_query("MATCH (s:Supplier) REMOVE s.s_nationkey;")

    def create_order_customer_relationships(self):
        print("Creating Customer -> Order relationships...")
        query = """
        MATCH (c:Customer), (o:Order {o_custkey: c.c_custkey})
        CREATE (c)-[:PLACED]->(o);
        """
        self.run_query(query)
        self.run_query("MATCH (o:Order) REMOVE o.o_custkey;")


def main():
    parser = argparse.ArgumentParser(description="Load TPC-H data into a Neo4j database.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the directory containing the TPC-H .tbl files. Files will be copied from there to the Neo4j import directory. If not specified, the files are expected to be already present in the Neo4j import directory."
    )
    parser.add_argument(
        "--neo4j-import-dir",
        type=str,
        default=None,
        help=(
            "Path to Neo4j's import directory. If not specified, reads from 'neo4j.import_dir' in config.yaml.\n"
            "Common locations:\n"
            "  - Linux (Debian/RPM): /var/lib/neo4j/import\n"
            "  - macOS (Homebrew):   /usr/local/var/neo4j/import or /opt/homebrew/var/neo4j/import\n"
            "  - Docker:             Mapped volume (often /import inside container)\n"
            "  - Neo4j Desktop:      Open App -> Manage -> Open Folder -> Import"
        )
    )
    parser.add_argument(
        "--reset-database",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to --no-reset-database to skip clearing the database beforehand."
    )

    args = parser.parse_args()

    # Load config from YAML
    config_file_path = 'config.yaml'
    config = {}
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return 
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return 

    neo4j_config = config.get('neo4j')
    if not neo4j_config:
        print("Error: 'neo4j' section not found in config.yaml")
        return
        
    NEO4J_URI = neo4j_config.get('uri')
    NEO4J_USER = neo4j_config.get('user')
    NEO4J_PASSWORD = neo4j_config.get('password')
    
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print("Error: 'uri', 'user', and 'password' must be defined under 'neo4j' in config.yaml")
        return
    
    # Get Neo4j import directory
    neo4j_import_dir = args.neo4j_import_dir or neo4j_config.get('import_dir')
    if not neo4j_import_dir:
        print("Error: Neo4j import directory must be specified via --neo4j-import-dir or 'import_dir' in config.yaml")
        return
    
    if not os.path.isdir(neo4j_import_dir):
        print(f"Error: Neo4j import directory does not exist: {neo4j_import_dir}")
        return

    # File Management
    tbl_files = ['region.tbl', 'nation.tbl', 'part.tbl', 'supplier.tbl', 
                 'customer.tbl', 'orders.tbl', 'partsupp.tbl', 'lineitem.tbl']
    copied_files = []
    
    if args.data_dir:
        # Copy files from data_dir to neo4j import directory
        if not os.path.isdir(args.data_dir):
            print(f"Error: Data directory does not exist: {args.data_dir}")
            return
        
        print(f"Copying .tbl files from '{args.data_dir}' to '{neo4j_import_dir}'...")
        for tbl_file in tbl_files:
            src = os.path.join(args.data_dir, tbl_file)
            dst = os.path.join(neo4j_import_dir, tbl_file)
            if not os.path.isfile(src):
                print(f"Error: Required file not found: {src}")
                return
            shutil.copy2(src, dst)
            copied_files.append(dst)
            print(f"  Copied: {tbl_file}")
    else:
        # Verify files exist in neo4j import directory
        print(f"Using .tbl files directly from Neo4j import directory: '{neo4j_import_dir}'")
        for tbl_file in tbl_files:
            filepath = os.path.join(neo4j_import_dir, tbl_file)
            if not os.path.isfile(filepath):
                print(f"Error: Required file not found in import directory: {filepath}")
                return
    # End File Management

    print(f"--- TPC-H Neo4j Loader ---")
    print(f"Reset database: {args.reset_database}")
    print(f"Connecting to Neo4j at: {NEO4J_URI}")
    print("----------------------------\n")

    loader = None
    try:
        loader = TpchLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        if args.reset_database:
            loader.reset_database()
        
        loader.create_constraints()
        loader.load_data()

        print("\nScript finished successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if loader:
            loader.close()
        
        # Clean up copied files
        if copied_files:
            print("\nCleaning up copied .tbl files from import directory...")
            for filepath in copied_files:
                try:
                    os.remove(filepath)
                    print(f"  Removed: {os.path.basename(filepath)}")
                except OSError as e:
                    print(f"  Warning: Could not remove {filepath}: {e}")

if __name__ == "__main__":
    main()