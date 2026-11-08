# Every implementation needs an observable contract, not just a happy path run. Empty input, invalid
# records, duplicates and mutation of caller data are the usual failure cases. Test what a function
# returns and also whether it modifies data the caller owns. Type hints help editors and static
# checkers but do not validate external data such as JSON. Use pytest or unittest when this grows
# into a larger project.


# Problem 1: quantity weighted average price.
#
# The function takes trades, a list of dicts that each have a numeric "price" and a numeric
# "quantity". It returns the average price weighted by quantity, which is the sum of price times
# quantity divided by the sum of quantity. A plain average of prices is wrong because a large trade
# counts more than a small one. If the total quantity is zero, which includes an empty list and a
# list where every quantity is zero, the function returns None. Trades with quantity zero do not
# change the result. The input is not modified.
#
# Example: [{"price": 10, "quantity": 1}, {"price": 20, "quantity": 3}] gives 17.5 because
# (10 + 60) / 4 = 17.5. An empty list gives None.
def weighted_average(trades: list[dict]) -> float | None:
    pass


if __name__ == '__main__':
    # Problem 2: executable checks for weighted_average.
    #
    # Add four plain assert statements. First, a trade with quantity zero next to a normal trade
    # gives the same average as the normal trade alone. Second, a single trade gives its own price.
    # Third, several trades give the quantity weighted value, for example 17.5 for the example above.
    # Fourth, an empty list gives None. The checks must run when the file is executed.
    pass
