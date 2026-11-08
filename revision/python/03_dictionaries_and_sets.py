
# Problem 1: total quantity per market.
#
# The function takes trades, a list of dicts. Each trade has a "market" key holding a string and a
# "quantity" key holding an int. It returns a dict mapping each market to the sum of the quantities
# of its trades. A market with trades appears once. An empty list gives an empty dict. A trade that
# lacks either key is invalid input and raises KeyError.
#
# Example: [{"market": "BTC", "quantity": 2}, {"market": "ETH", "quantity": 1},
# {"market": "BTC", "quantity": 3}] gives {"BTC": 5, "ETH": 1}.
#
# Aim for one pass over the list in O(n) using dict.get with a default of zero.
def count_by_market(trades: list[dict]) -> dict[str, int]:
    res = {}
    for trade in trades:
        market = trade["market"]
        res[market] = res.get(market, 0) + trade["quantity"]
    return res


# Problem 2: remove duplicates and keep order.
#
# The function takes items, a list of strings, and returns a new list holding each distinct string
# once, in the order of its first appearance. An empty list gives an empty list. The input is not
# modified.
#
# Example: ["a", "b", "a", "c", "b"] gives ["a", "b", "c"].
#
# Aim for O(n) time with a set of items already seen.
def unique_in_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    
    for item in items:
      if item not in seen:
        result.append(item)
        seen.add(item)
    
    return result


# Problem 3: first repeated item.
#
# The function takes items, a list of strings, and returns the first item that is encountered for the
# second time while reading the list from left to right. If no item repeats, including for an empty
# list, it returns None. The answer is decided by the position of the second occurrence, not the
# first.
#
# Example: ["a", "b", "b", "a"] gives "b" because the second "b" arrives before the second "a".
# ["a", "b", "c"] gives None.
#
# Aim for O(n) time with a set.
def first_repeated(items: list[str]) -> str | None:
    # return the first item encountered for the second time, or None.
    seen = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None
