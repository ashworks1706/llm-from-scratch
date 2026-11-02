# every implementation needs an observable contract, not just a happy-path run.
# empty input, invalid records, duplicates and mutation are frequent failure cases.
# test what a function returns AND whether it modifies caller-owned data.
# typing helps editors and static checkers; it does not validate external JSON.
# use pytest or standard unittest when extending this into a larger project.


def weighted_average(trades: list[dict]) -> float | None:
    # sum(price * quantity) / sum(quantity); zero total quantity -> None.
    # do not average prices without considering quantities.
    pass


if __name__ == '__main__':
    # after implementing, add assertions for zero quantities, one trade,
    # multiple trades and an empty list. keep the checks executable.
    pass
