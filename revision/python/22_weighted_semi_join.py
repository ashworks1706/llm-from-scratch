# when units carry their own weight and members are filtered in a child table, a plain join repeats the unit weight once per matching member.
# filtering through a distinct set of unit ids, a semi join, keeps each unit weight counted exactly once.
# the filtered weighted total can never be larger than the weighted total with no member filter.

import sqlite3


def build_schema(conn: sqlite3.Connection) -> None:
    # create units(unit_id primary key, size, weight) and members(unit_id, age, enrolled) with an index on members(unit_id).
    # a unit may have zero members, and every members.unit_id must exist in units.
    pass


def naive_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    # return sum(units.weight) using an inner join to members filtered by age >= min_age.
    # this version is expected to overcount units with several qualifying members.
    pass


def semi_join_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    # return sum(units.weight) for units with at least one member aged min_age or older, using exists or in over distinct ids.
    # use bound parameters, never string formatting, and return 0 rather than None when nothing matches.
    pass


if __name__ == '__main__':
    # build an in memory database with one unit of weight 10 and three adult members.
    # assert the naive total is 30 and the semi join total is 10.
    # assert semi_join_weighted_total returns 0 when no member qualifies.
    pass
