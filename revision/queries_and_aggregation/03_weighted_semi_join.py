# joining units to filtered members repeats each unit weight per matching member. a semi join on
# distinct unit ids counts each weight once; the filtered total can't exceed the unfiltered one.

import sqlite3


# 1. build_schema
# create units(unit_id primary key, size, weight) and members(unit_id, age, enrolled) where
# members.unit_id is a foreign key to units, plus an index on members(unit_id). returns nothing.
# ex: a unit with zero members inserts fine; a member with an unknown unit_id errors when foreign
#     keys are enforced
def build_schema(conn: sqlite3.Connection) -> None:
    pass


# 2. naive_weighted_total
# sum(units.weight) over an inner join to members with age >= min_age. counts a unit once per
# matching member (the overcount is the point). nothing matches -> 0, not None.
# ex: one unit (weight 10) with members aged 30, 40, 50, min_age 18 -> 30
def naive_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


# 3. semi_join_weighted_total
# sum(units.weight) over units with at least one member aged >= min_age, each unit once.
# use EXISTS or IN over distinct qualifying unit ids. bind min_age, no string formatting.
# none qualify -> 0, not None.
# ex: same unit -> 10 for min_age 18, 0 for min_age 99
def semi_join_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


if __name__ == '__main__':
    # in-memory db with the schema, one unit of weight 10 and three adult members.
    # check 1: naive total 30, semi join total 10.
    # check 2: semi_join_weighted_total -> 0 when no member qualifies.
    pass
