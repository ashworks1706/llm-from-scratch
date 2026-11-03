
def count_by_market(trades: list[dict]) -> dict[str, int]:
    res = {}
    for trade in trades:
        market = trade["market"]
        res[market] += res.get(market, 0) + trade["quantity"]
    return res


def unique_in_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    
    for item in items:
      if item not in seen:
        result.append(item)
        seen.add(item)
    
    return result


def first_repeated(items: list[str]) -> str | None:
    # return the first item encountered for the second time, or None.
    seen = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None
