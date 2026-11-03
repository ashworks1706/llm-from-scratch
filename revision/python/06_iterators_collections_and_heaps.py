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

def most_common_market(markets: list[str]) -> str | None:
    if not markets:
        return None

    c = Counter(markets)
    return c.most_common(1)[0][0]


def cheapest_k(asks: list[dict], k: int) -> list[dict]:
    sor = sorted(asks, key=lambda x: x["price"])
    return sor[:max(0, k)]
