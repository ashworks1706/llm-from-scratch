# A comprehension transforms or filters an iterable in one expression, and a generator expression
# does the same lazily. sorted(data, key=f) orders items by f(item), not by the item itself, and
# max(records, key=f) returns the whole record with the greatest f(record). list.sort() mutates the
# list and returns None, while sorted() returns a new list. Use a tuple as the key to order by
# several columns.


# Problem 1: sort orders by quantity then price.
#
# The function takes orders, a list of dicts that each have a numeric "quantity" and a numeric
# "price". It returns a NEW list holding the same dicts ordered by quantity from largest to smallest,
# and among equal quantities by price from smallest to largest. Orders that tie on both keep their
# original relative order. The input list is not modified. An empty list gives an empty list.
#
# Example: [{"quantity": 1, "price": 5}, {"quantity": 3, "price": 9}, {"quantity": 3, "price": 2}]
# gives the order quantity 3 price 2, then quantity 3 price 9, then quantity 1 price 5.
#
# Aim for one call to sorted with a tuple key.
def descending_by_quantity(orders: list[dict]) -> list[dict]:
    pass


# Problem 2: the most valuable order.
#
# The function takes orders, a list of dicts that each have a numeric "price" and a numeric
# "quantity". The value of an order is price times quantity. It returns the whole order dict with the
# greatest value. If several orders tie, return the first of them in the list. An empty list returns
# None instead of letting max() raise.
#
# Example: [{"price": 2, "quantity": 5}, {"price": 4, "quantity": 3}] gives the second order because
# 12 is larger than 10. An empty list gives None.
def most_valuable_order(orders: list[dict]) -> dict | None:
    pass


# Problem 3: prices of sufficiently large orders.
#
# The function takes orders, a list of dicts with a numeric "price" and an int "quantity", and
# minimum, an int. It returns a list of the prices of the orders whose quantity is greater than or
# equal to minimum, in the original order of the list. If none qualify, or the list is empty, it
# returns an empty list.
#
# Example: with orders of (price 1.5, quantity 2), (price 3.0, quantity 5) and (price 2.5, quantity 5)
# and minimum 5, the result is [3.0, 2.5].
#
# Aim for a single list comprehension.
def active_prices(orders: list[dict], minimum: int) -> list[float]:
    pass
