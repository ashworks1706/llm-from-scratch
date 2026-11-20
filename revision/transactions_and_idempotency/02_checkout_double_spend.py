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
def price_items(db: sqlite3.Connection, items: list[tuple[int, int]]) -> float:
    pass


# 2. place_order
# BEGIN IMMEDIATE, then: price from catalog, check balance >= total, decrement each product's stock,
# insert an orders row, insert a ledger row of -total. return the amount charged.
# any failure (low balance, StoreError from pricing, db error) -> roll back everything and re-raise:
# no stock moves, no ledger row. total always comes from the catalog.
def place_order(db: sqlite3.Connection, user_id: int, items: list[tuple[int, int]]) -> float:
    pass


# ---- try it ----
if __name__ == '__main__':
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.executescript("create table products(id integer primary key, price real, stock integer);"
                     "create table ledger(user_id integer, amount real);"
                     "create table orders(id integer primary key, user_id integer, total real);"
                     "insert into products values (1, 10, 5); insert into ledger values (7, 100);")
    # assert abs(price_items(db, [(1, 2), (1, 3)]) - 50.0) < 1e-6
    # price_items(db, [(1, 3), (1, 3)])             # raises StoreError (6 > 5)
    # price_items(db, [(99, 1)])                    # raises StoreError
    # assert place_order(db, 7, [(1, 3)]) == 30.0  # stock 5 -> 2, balance 100 -> 70
    # place_order(db, 8, [(1, 1)])                  # raises StoreError: user 8 has no balance, nothing changes
    pass
