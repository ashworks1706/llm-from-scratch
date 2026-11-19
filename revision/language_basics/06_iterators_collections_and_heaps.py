from collections import Counter, defaultdict, deque
import heapq


# 1. most_common_market
# return the most frequent name; ties -> any of them. [] -> None.
# ex: ["BTC", "ETH", "BTC", "SOL", "ETH", "BTC"] -> "BTC", [] -> None
def most_common_market(markets: list[str]) -> str | None:
    if not markets:
        return None

    c = Counter(markets)
    return c.most_common(1)[0][0]


# 2. cheapest_k
# asks: dicts with numeric "price". return the k lowest-priced asks, cheapest first.
# input unsorted, don't mutate it. equal prices keep original order. k <= 0 -> [], k > len -> all.
# ex: asks priced 0.70, 0.40, 0.60, 0.30 with k=2 -> [the 0.30 ask, the 0.40 ask]
# O(n log k) with a size-k heap, don't sort the whole list.
def cheapest_k(asks: list[dict], k: int) -> list[dict]:
    sor = sorted(asks, key=lambda x: x["price"])
    return sor[:max(0, k)]
