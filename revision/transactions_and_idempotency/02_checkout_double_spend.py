# checkout commits stock, order and ledger together or not at all. charge the catalog total, never a
# client total. read-then-write balance checks race. stock is checked cumulatively per product.
# tables (caller creates): products(id, price, stock), ledger(user_id, amount) where balance =
# sum(amount), orders(id, user_id, total).

import sqlite3


# raised for any checkout failure: unknown product, not enough stock, or not enough balance.
class StoreError(Exception):
    pass


# 1. price_items
# items: (product_id, qty >= 1). return the catalog total sum(price * qty) as a float.
# a product may repeat across lines; sum its qtys before comparing with stock.
# unknown product id or cumulative qty > stock -> StoreError. never changes the db.
# ex: product 1 (price 10, stock 5): [(1, 2), (1, 3)] -> 50.0
#     [(1, 3), (1, 3)] -> StoreError (6 > 5), [(99, 1)] -> StoreError
def price_items(db: sqlite3.Connection, items: list[tuple[int, int]]) -> float:
    pass


# 2. place_order
# BEGIN IMMEDIATE, then: price from catalog, check balance >= total, decrement each product's stock,
# insert an orders row, insert a ledger row of -total. return the amount charged.
# any failure (low balance, StoreError from pricing, db error) -> roll back everything and re-raise:
# no stock moves, no ledger row. total always comes from the catalog.
# ex: balance 100 buying one 30-point item -> 30.0, stock -1, balance 70.
#     balance 20 buying it -> StoreError, nothing changes
def place_order(db: sqlite3.Connection, user_id: int, items: list[tuple[int, int]]) -> float:
    pass


if __name__ == '__main__':
    # check 1: order total above balance -> StoreError, every product's stock unchanged.
    # check 2: 8 threads each buy one 100-point item with balance 100 -> exactly one succeeds,
    # seven fail, final balance 0.
    pass
