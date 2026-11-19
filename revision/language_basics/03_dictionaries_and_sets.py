# 1. count_by_market
# trades: list of {"market": str, "quantity": int}. return {market: total quantity}.
# [] -> {}. a trade missing a key raises KeyError.
# ex: [{"market": "BTC", "quantity": 2}, {"market": "ETH", "quantity": 1}, {"market": "BTC", "quantity": 3}]
#     -> {"BTC": 5, "ETH": 1}
# O(n), one pass with dict.get(k, 0).
def count_by_market(trades: list[dict]) -> dict[str, int]:
    res = {}
    for trade in trades:
        market = trade["market"]
        res[market] = res.get(market, 0) + trade["quantity"]
    return res


# 2. unique_in_order
# return each distinct string once, in first-seen order. don't mutate the input.
# ex: ["a", "b", "a", "c", "b"] -> ["a", "b", "c"]
# O(n) with a seen set.
def unique_in_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    
    for item in items:
      if item not in seen:
        result.append(item)
        seen.add(item)
    
    return result


# 3. first_repeated
# return the first item seen for the second time, else None ([] -> None).
# decided by where the second occurrence lands, not the first.
# ex: ["a", "b", "b", "a"] -> "b",  ["a", "b", "c"] -> None
# O(n) with a set.
def first_repeated(items: list[str]) -> str | None:
    # return the first item encountered for the second time, or None.
    seen = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None
