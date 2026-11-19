

# 1. descending_by_quantity
# orders: dicts with numeric "quantity" and "price". return a NEW list sorted by quantity desc,
# then price asc; full ties keep original order. input unchanged. [] -> [].
# ex: [{q: 1, p: 5}, {q: 3, p: 9}, {q: 3, p: 2}] -> (q3, p2), (q3, p9), (q1, p5)
# one sorted() call with a tuple key.
def descending_by_quantity(orders):
    return sorted(
        orders,
        key=lambda order: (-order["quantity"], order["price"])
    )

      


# 2. most_valuable_order
# value = price * quantity. return the whole order dict with the max value; ties -> first in list.
# [] -> None (don't let max() raise).
# ex: [{"price": 2, "quantity": 5}, {"price": 4, "quantity": 3}] -> second (12 > 10)
def most_valuable_order(orders):
    if not orders:
        return None

    return max(orders, key=lambda x: x["price"] * x["quantity"])

# 3. active_prices
# orders: dicts with "price" and int "quantity". return prices of orders with quantity >= minimum,
# in original order. none qualify or [] -> [].
# ex: (price 1.5, q 2), (3.0, q 5), (2.5, q 5) with minimum 5 -> [3.0, 2.5]
# single list comprehension.
def active_prices(orders, minimum):
    return [ order["price"] for order in orders if order["quantity"] >= minimum]
