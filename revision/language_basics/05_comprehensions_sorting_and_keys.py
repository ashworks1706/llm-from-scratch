

# 1. descending_by_quantity
# orders: dicts with numeric "quantity" and "price". return a NEW list sorted by quantity desc,
# then price asc; full ties keep original order. input unchanged. [] -> [].
# one sorted() call with a tuple key.
def descending_by_quantity(orders):
    return sorted(
        orders,
        key=lambda order: (-order["quantity"], order["price"])
    )

      


# 2. most_valuable_order
# value = price * quantity. return the whole order dict with the max value; ties -> first in list.
# [] -> None (don't let max() raise).
def most_valuable_order(orders):
    if not orders:
        return None

    return max(orders, key=lambda x: x["price"] * x["quantity"])

# 3. active_prices
# orders: dicts with "price" and int "quantity". return prices of orders with quantity >= minimum,
# in original order. none qualify or [] -> [].
# single list comprehension.
def active_prices(orders, minimum):
    return [ order["price"] for order in orders if order["quantity"] >= minimum]


# ---- try it ----
if __name__ == "__main__":
    # orders = [{"quantity": 1, "price": 5}, {"quantity": 3, "price": 9}, {"quantity": 3, "price": 2}]
    # print(descending_by_quantity(orders))         # (q3, p2), (q3, p9), (q1, p5)
    # assert most_valuable_order([{"price": 2, "quantity": 5}, {"price": 4, "quantity": 3}]) == {'price': 4, 'quantity': 3}
    # assert active_prices([{"price": 1.5, "quantity": 2}, {"price": 3.0, "quantity": 5}, {"price": 2.5, "quantity": 5}], 5) == [3.0, 2.5]
    pass
