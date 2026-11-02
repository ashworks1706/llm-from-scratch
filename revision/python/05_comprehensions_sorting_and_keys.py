# a comprehension transforms or filters an iterable; generators produce lazily.
# sorted(data, key=...) compares the result of key(item), not item itself.
# max(records, key=...) returns the full record with greatest derived score.
# list.sort() mutates the original and returns None; sorted() returns a copy.
# use tuple-valued sort keys for multi-column ordering.


def descending_by_quantity(orders: list[dict]) -> list[dict]:
    # return a copy sorted by quantity descending, then price ascending.
    pass


def most_valuable_order(orders: list[dict]) -> dict | None:
    # an order has price and quantity; compare price * quantity.
    # empty input returns None rather than letting max() raise.
    pass


def active_prices(orders: list[dict], minimum: int) -> list[float]:
    # return prices for orders whose quantity >= minimum.
    pass
