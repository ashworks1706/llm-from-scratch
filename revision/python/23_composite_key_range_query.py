# A fact table keyed by (metric_code, place_name, start_year, end_year) answers point and period
# lookups through its composite primary key. The leftmost key columns must match the equality
# filters used most often, or the index cannot be used. Year ranges supplied by a user are validated
# at the boundary and rejected with a clear error instead of quietly returning no data. Every filter
# value travels as a bound parameter, so one query shape is reused for any selection.
#
# The table is called facts and has the columns metric_code, place_name, start_year, end_year and
# value, with the four key columns as its primary key.

import re
import sqlite3

YEAR_RANGE = re.compile(r"^(\d{4})(?:-(\d{4}))?$")


# Problem 1: parse year ranges.
#
# The function takes values, a list of strings or None. Each string is either one four digit year
# such as "2020", which means the pair (2020, 2020), or two years joined by a hyphen such as
# "2018-2022", which means (2018, 2022). It returns the list of pairs in input order. None or an
# empty list gives an empty list. Text that does not match one of the two forms, such as "20x1" or
# "2020-", raises ValueError, and so does a range whose end year is before its start year.
#
# Example: ["2020", "2018-2022"] gives [(2020, 2020), (2018, 2022)]. "2022-2018" raises ValueError.
# "20x1" raises ValueError.
def parse_year_ranges(values: list[str] | None) -> list[tuple[int, int]]:
    pass


# Problem 2: build the query.
#
# The function takes metric_codes, a list of strings, place_names, a list of strings, and ranges, a
# list of (start_year, end_year) pairs. It returns a tuple of the SQL text and the list of parameters,
# a SELECT over facts with a WHERE clause joined by AND. The clause for metric_code is an IN list
# with one question mark placeholder per code, the clause for place_name likewise, and the clause for
# ranges is the OR of one (start_year = ? AND end_year = ?) group per pair, wrapped in parentheses.
# A list that is empty contributes no clause at all, and if all three are empty there is no WHERE.
# The parameters come in the same order as the placeholders, with no values written into the SQL
# text.
#
# Example: build_query(["A", "B"], [], []) gives SQL whose only filter is metric_code IN (?, ?) and
# the parameters ["A", "B"]. Adding the range (2020, 2020) appends the group
# (start_year = ? AND end_year = ?) and the parameters 2020 and 2020.
def build_query(metric_codes: list[str], place_names: list[str], ranges: list[tuple[int, int]]) -> tuple[str, list]:
    pass


# Problem 3: does the query use the primary key.
#
# The function takes conn, an open connection, sql, a query string, and params, its parameter list.
# It runs EXPLAIN QUERY PLAN on the query and returns True when the plan shows a SEARCH, meaning the
# primary key or another index is used, and False when it shows a full table SCAN. The query itself
# is not executed.
#
# Example: for a table created WITHOUT ROWID with primary key (metric_code, place_name, start_year,
# end_year), a query filtered on metric_code = ? gives True, and a query filtered only on value gives
# False.
def uses_primary_key(conn: sqlite3.Connection, sql: str, params: list) -> bool:
    pass


if __name__ == '__main__':
    # Check one: parse_year_ranges rejects "2022-2018" and "20x1" with ValueError.
    # Check two: build_query with only metric codes produces one IN clause and exactly as many
    # parameters as codes.
    # Check three: uses_primary_key is True for a metric code filter on a WITHOUT ROWID table.
    pass
