

def descending_by_quantity(orders):
    return sorted(
        orders,
        key=lambda order: (-order["quantity"], order["price"])
    )

      


def most_valuable_order(orders):
    if not orders:
        return None

    return max(orders, key=lambda x: x["price"] * x["quantity"])

def active_prices(orders, minimum):
    return [ order["price"] for order in orders if order["quantity"] >= minimum]
