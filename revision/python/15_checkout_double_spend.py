# checkout changes stock, the order and a negative ledger row together, so it commits once or not at all.
# the charge is the catalog total, never a total the client sent.
# a balance check that reads and then writes is a check-then-act race, so two requests both see enough.
# stock is checked cumulatively per product because two lines of one product add up.

import sqlite3


class StoreError(Exception):
    pass


def price_items(db: sqlite3.Connection, items: list[tuple[int, int]]) -> float:
    # items are (product_id, qty) with qty >= 1.
    # raise StoreError for an unknown product or cumulative qty above stock, change nothing, and return the catalog total.
    pass


def place_order(db: sqlite3.Connection, user_id: int, items: list[tuple[int, int]]) -> float:
    # inside BEGIN IMMEDIATE, price the items, check balance >= total, decrement stock, and insert a ledger row of -total.
    # any failure rolls everything back. return the charge.
    pass


if __name__ == '__main__':
    # assert that an insufficient balance changes no stock.
    # run eight threads buying a 100 point item with a 100 point balance and assert exactly one succeeds.
    pass
