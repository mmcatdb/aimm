from query_engine import QueryEngine


def main():
    engine = QueryEngine()

    try:
        print("--- Finding a customer ---")
        customer = engine.find('customer', {'c_custkey': 1})
        print(customer)
        print("")
        

        print("--- Finding number of orders with status F ---")
        orders = engine.find('orders', {'o_orderstatus': 'F'})
        print(f"Found {len(orders)} orders with status 'F'")
        print("")
        

        print("--- Lineitems for a customer ---")
        lineitems = engine.find_lineitems_for_customer('Customer#000000001')
        print("--- Results ---")
        for item in lineitems:
            print(item)

    finally:
        engine.disconnect_all()
        print("\nDisconnected from all databases.")


if __name__ == "__main__":
    main()