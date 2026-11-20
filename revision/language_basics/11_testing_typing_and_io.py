# test the contract, not just the happy path: empty input, bad records, duplicates, caller mutation.
# type hints don't validate external data such as JSON.


# 1. weighted_average
# trades: dicts with numeric "price" and "quantity". return sum(price * quantity) / sum(quantity).
# total quantity 0 (incl. [] and all-zero quantities) -> None. quantity-0 trades don't change it.
# input unchanged.
def weighted_average(trades: list[dict]) -> float | None:
    pass


# ---- try it ----
if __name__ == '__main__':
    # assert abs(weighted_average([{"price": 10, "quantity": 1}, {"price": 20, "quantity": 3}]) - 17.5) < 1e-6
    # assert weighted_average([]) is None
    pass
