# a dictionary is a hash table from hashable keys to values.
# dict[key] raises KeyError if missing; dict.get(key, default) does not.
# sets track unique elements; membership is O(1) on average.
# don't use an entire dictionary as a key: dictionaries are unhashable.
# most accumulation bugs come from reading totals[key] before it exists.


def count_by_market(trades: list[dict]) -> dict[str, int]:
    # each record has 'market' and 'quantity'. accumulate quantities by market.
    # the value added is trade['quantity'], not the entire record.
    pass


def unique_in_order(items: list[str]) -> list[str]:
    # preserve first appearance; use a set for efficient membership.
    pass


def first_repeated(items: list[str]) -> str | None:
    # return the first item encountered for the second time, or None.
    pass
