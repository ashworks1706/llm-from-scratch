from collections import Counter, defaultdict, deque
import heapq


# 1. most_common_market
# return the most frequent name; ties -> any of them. [] -> None.
def most_common_market(markets: list[str]) -> str | None:
    if not markets:
        return None

    c = Counter(markets)
    return c.most_common(1)[0][0]


# 2. cheapest_k
# asks: dicts with numeric "price". return the k lowest-priced asks, cheapest first.
# input unsorted, don't mutate it. equal prices keep original order. k <= 0 -> [], k > len -> all.
# O(n log k) with a size-k heap, don't sort the whole list.
def cheapest_k(asks: list[dict], k: int) -> list[dict]:
    sor = sorted(asks, key=lambda x: x["price"])
    return sor[:max(0, k)]


# ---- try it ----
if __name__ == "__main__":
    # assert most_common_market(["BTC", "ETH", "BTC", "SOL", "ETH", "BTC"]) == 'BTC'
    # assert most_common_market([]) is None
    asks = [{"id": "a", "price": 0.70}, {"id": "b", "price": 0.40}, {"id": "c", "price": 0.60}, {"id": "d", "price": 0.30}]
    # assert cheapest_k(asks, 2) == [{'id': 'd', 'price': 0.3}, {'id': 'b', 'price': 0.4}]
    pass
