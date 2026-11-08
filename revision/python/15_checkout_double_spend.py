# Checkout changes stock, the order and a negative ledger row together, so it commits once or not at
# all. The charge is the catalog total, never a total the client sent. A balance check that reads and
# then writes is a check then act race, so two requests can both see enough. Stock is checked
# cumulatively per product, because two lines for the same product add up.
#
# The database these functions use has these tables, created by the caller. products(id, price,
# stock) holds the catalog. ledger(user_id, amount) holds signed point movements, and a user's
# balance is the sum of their amounts. orders(id, user_id, total) records each purchase.

import sqlite3


# Raised for any checkout failure: unknown product, not enough stock, or not enough balance.
class StoreError(Exception):
    pass


# Problem 1: price a basket.
#
# The function takes db, a connection, and items, a list of (product_id, qty) tuples where qty is at
# least 1. It returns the total price from the catalog, the sum of price times qty over all lines,
# as a float. The same product may appear on several lines, and the quantities of those lines are
# added before comparing with stock. If any product id is not in the catalog, or the cumulative qty
# for a product is above its stock, it raises StoreError. It never changes the database, whether it
# succeeds or fails.
#
# Example: with product 1 priced 10 and stock 5, items [(1, 2), (1, 3)] give 50.0, while
# [(1, 3), (1, 3)] raises StoreError because 6 is more than 5. [(99, 1)] raises StoreError.
def price_items(db: sqlite3.Connection, items: list[tuple[int, int]]) -> float:
    pass


# Problem 2: place an order.
#
# The function takes db, user_id, an int, and items, a list of (product_id, qty) tuples. It starts
# a transaction with BEGIN IMMEDIATE so that no other writer can slip in between the checks and the
# writes. Inside it prices the items with the catalog, checks that the user's balance is at least
# the total, decrements the stock of each product, inserts an order row, and inserts a ledger row
# of minus total for the user. It returns the amount charged. If anything fails, including an
# insufficient balance, a StoreError from pricing, or a database error, every change is rolled back
# and the error reaches the caller, so no stock moves and no ledger row appears. The total always
# comes from the catalog, never from anything the caller sends.
#
# Example: a user with balance 100 buying one 30 point item is charged 30.0, stock drops by one,
# and the balance becomes 70. A user with balance 20 buying that item gets StoreError and nothing
# changes.
def place_order(db: sqlite3.Connection, user_id: int, items: list[tuple[int, int]]) -> float:
    pass


if __name__ == '__main__':
    # Check one: an order whose total is above the user's balance raises StoreError and leaves every
    # product's stock unchanged.
    # Check two: eight threads each try to buy one 100 point item with a balance of 100 points, and
    # exactly one of them succeeds. The other seven fail, and the final balance is zero.
    pass
