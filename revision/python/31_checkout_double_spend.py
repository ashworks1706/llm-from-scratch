# checkout touches three things that must change together: stock, the order, and a negative ledger row.
# one commit, or none. the charge is the catalog total, never the total the client sent.
# stock is checked cumulatively per product (two lines of the same product add up).
# a balance check that reads and then writes is a check-then-act race: two requests both see enough.
# from platform: modules/storefront/api.py place_order() and modules/storefront/service.py price_items(),
# from platform: take_stock(), create_order(charge=...) which adds Points(points=-charge) in the same commit.
# from platform: price_items locks Product rows (with_for_update); the balance SUM itself is not locked.
# explain: which concurrent checkouts does a lock on product rows serialize, and which does it not?
# explain: name two ways to close the user-level race (lock a user row, or a guarded insert).

import sqlite3


class StoreError(Exception):
    pass


def make_db(balance: float, stock: dict[int, int], prices: dict[int, float]) -> sqlite3.Connection:
    # products(id, stock, price), ledger(user_id, points). one user (id 1) with a starting top-up row.
    # check_same_thread=False so threads can share it in the race test.
    pass


def price_items(db: sqlite3.Connection, items: list[tuple[int, int]]) -> float:
    # items are (product_id, qty). qty >= 1. unknown product or cumulative qty above stock -> StoreError.
    # return the catalog total. change nothing.
    pass


def place_order(db: sqlite3.Connection, user_id: int, items: list[tuple[int, int]]) -> float:
    # inside BEGIN IMMEDIATE (the toy lock): price, check balance >= total, decrement stock,
    # insert ledger row -total. any failure rolls everything back. return the charge.
    pass


if __name__ == '__main__':
    # assert: insufficient balance changes no stock, duplicate lines of one product respect stock,
    # then run 8 threads buying one 100-point item with a 100-point balance: exactly one succeeds.
    # remove the lock and watch the balance go negative. that is the bug to explain.
    pass
