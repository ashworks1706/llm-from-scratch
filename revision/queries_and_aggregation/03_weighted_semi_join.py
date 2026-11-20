# joining units to filtered members repeats each unit weight per matching member. a semi join on
# distinct unit ids counts each weight once; the filtered total can't exceed the unfiltered one.

import sqlite3


# 1. build_schema
# create units(unit_id primary key, size, weight) and members(unit_id, age, enrolled) where
# members.unit_id is a foreign key to units, plus an index on members(unit_id).
def build_schema(conn: sqlite3.Connection) -> None:
    pass


# 2. naive_weighted_total
# sum(units.weight) over an inner join to members with age >= min_age. counts a unit once per
# matching member (the overcount is the point). nothing matches -> 0, not None.
def naive_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


# 3. semi_join_weighted_total
# sum(units.weight) over units with at least one member aged >= min_age, each unit once.
# use EXISTS or IN over distinct qualifying unit ids. bind min_age, no string formatting.
# none qualify -> 0, not None.
def semi_join_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


# ---- try it ----
if __name__ == '__main__':
    # conn = sqlite3.connect(":memory:")
    # build_schema(conn)
    # conn.execute("insert into units values (1, 3, 10)")
    # conn.executemany("insert into members values (1, ?, 1)", [(30,), (40,), (50,)])
    # assert naive_weighted_total(conn, 18) == 30
    # assert semi_join_weighted_total(conn, 18) == 10
    # assert semi_join_weighted_total(conn, 99) == 0
    # conn.execute("insert into units values (2, 1, 5)")   # ok: a unit with no members
    # conn.execute("pragma foreign_keys = on")
    # conn.execute("insert into members values (99, 30, 1)")  # raises sqlite3.IntegrityError: unknown unit_id
    pass
