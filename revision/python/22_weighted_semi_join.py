# When units carry their own weight and members are filtered in a child table, a plain join repeats
# the unit weight once per matching member. Filtering through a distinct set of unit ids, a semi
# join, keeps each unit weight counted exactly once. The filtered weighted total can never be larger
# than the weighted total with no member filter.

import sqlite3


# Problem 1: create the schema.
#
# The function takes conn, an open sqlite3 connection, and creates two tables. units has the columns
# unit_id (primary key), size and weight. members has the columns unit_id, age and enrolled, where
# every unit_id must refer to an existing unit, so declare it as a foreign key. Also create an index
# on members(unit_id). A unit may have zero members. The function returns nothing.
#
# Example: after the call, a unit can be inserted with no members, but inserting a member whose
# unit_id has no unit is an error when foreign keys are enforced.
def build_schema(conn: sqlite3.Connection) -> None:
    pass


# Problem 2: the naive weighted total.
#
# The function takes conn and min_age, an int. It returns the sum of units.weight using an inner join
# from units to members restricted to members with age >= min_age. Because the join produces one row
# per matching member, a unit with several qualifying members is counted several times. This version
# is meant to show that overcount. When nothing matches, return 0 rather than None.
#
# Example: one unit of weight 10 with three members aged 30, 40 and 50, and min_age 18, gives 30.
def naive_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


# Problem 3: the weighted total through a semi join.
#
# The function takes conn and min_age, an int. It returns the sum of units.weight over the units
# that have at least one member aged min_age or older, counting each such unit once. Use EXISTS, or
# IN over the distinct unit ids of the qualifying members. Pass min_age as a bound parameter and do
# not build the SQL with string formatting. When no unit qualifies, return 0 rather than None.
#
# Example: the same unit of weight 10 with three adult members gives 10 for min_age 18. With min_age
# 99 nobody qualifies and the result is 0.
def semi_join_weighted_total(conn: sqlite3.Connection, min_age: int) -> float:
    pass


if __name__ == '__main__':
    # Build an in memory database with the schema, one unit of weight 10 and three adult members.
    # Check one: the naive total is 30 and the semi join total is 10.
    # Check two: semi_join_weighted_total returns 0 when no member qualifies.
    pass
