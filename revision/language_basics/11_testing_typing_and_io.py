# test the contract, not just the happy path: empty input, bad records, duplicates, caller mutation.
# type hints don't validate external data such as JSON.


# 1. weighted_average
# trades: dicts with numeric "price" and "quantity". return sum(price * quantity) / sum(quantity).
# total quantity 0 (incl. [] and all-zero quantities) -> None. quantity-0 trades don't change it.
# input unchanged.
# ex: [{"price": 10, "quantity": 1}, {"price": 20, "quantity": 3}] -> 17.5, [] -> None
def weighted_average(trades: list[dict]) -> float | None:
    pass


if __name__ == '__main__':
    # 2. four plain asserts on weighted_average, run when the file executes:
    # quantity-0 trade beside a normal one == normal one alone; single trade -> its own price;
    # several trades -> weighted value (17.5 for the example); [] -> None.
    pass
