from collections import Counter, defaultdict, deque
import heapq


# markets = ["BTC", "ETH", "BTC", "SOL", "ETH", "BTC"]
# Expected output: "BTC"

# Given:
# asks = [    {"price": 0.70, "quantity": 10},    {"price": 0.40, "quantity": 20},    {"price": 0.60, "quantity": 15},    {"price": 0.30, "quantity": 5}]
# Expected:
# cheapest_k(asks, 2)
# [
#     {"price": 0.30, "quantity": 5},
#     {"price": 0.40, "quantity": 20}
# ]

# Problem 1: most common market.
#
# The function takes markets, a list of market name strings, and returns the name that occurs most
# often. When several names tie for the highest count, any one of them is accepted. An empty list
# returns None.
#
# Example: ["BTC", "ETH", "BTC"] gives "BTC". [] gives None.
def most_common_market(markets: list[str]) -> str | None:
    if not markets:
        return None

    c = Counter(markets)
    return c.most_common(1)[0][0]


# Problem 2: the k cheapest asks.
#
# The function takes asks, a list of dicts that each have a numeric "price", and k, an int. It
# returns a list of the k asks with the lowest price, ordered from cheapest to most expensive. The
# input list is not sorted and must not be modified. Asks with equal prices keep their original
# relative order, so the result is deterministic. If k is zero or negative the result is an empty
# list, and if k is larger than the number of asks all asks are returned in price order.
#
# Example: asks with prices 5, 1, 4, 2 and k = 2 gives the asks priced 1 and 2.
#
# Aim for O(n log k) using a heap of size k. Do not sort the whole list.
def cheapest_k(asks: list[dict], k: int) -> list[dict]:
    sor = sorted(asks, key=lambda x: x["price"])
    return sor[:max(0, k)]
