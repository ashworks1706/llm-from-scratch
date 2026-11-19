# leftmost composite-key columns must match the common equality filters or the index is unused.
# validate user year ranges at the boundary (clear error, not empty results); bind every value.
# table facts(metric_code, place_name, start_year, end_year, value), pk = the first four columns.

import re
import sqlite3

YEAR_RANGE = re.compile(r"^(\d{4})(?:-(\d{4}))?$")


# 1. parse_year_ranges
# each string is "YYYY" -> (y, y) or "YYYY-YYYY" -> (start, end). return pairs in input order.
# None or [] -> []. any other form ("20x1", "2020-") -> ValueError. end < start -> ValueError.
# ex: ["2020", "2018-2022"] -> [(2020, 2020), (2018, 2022)], "2022-2018" -> ValueError
def parse_year_ranges(values: list[str] | None) -> list[tuple[int, int]]:
    pass


# 2. build_query
# return (sql, params): SELECT over facts with WHERE clauses joined by AND:
# metric_code IN (?, ...) one ? per code; place_name IN (...) likewise; ranges as the OR of one
# (start_year = ? AND end_year = ?) per pair, wrapped in parentheses.
# an empty list adds no clause; all empty -> no WHERE. params in placeholder order, none in sql.
# ex: (["A", "B"], [], []) -> only filter metric_code IN (?, ?), params ["A", "B"].
#     adding range (2020, 2020) appends (start_year = ? AND end_year = ?) and params 2020, 2020
def build_query(metric_codes: list[str], place_names: list[str], ranges: list[tuple[int, int]]) -> tuple[str, list]:
    pass


# 3. uses_primary_key
# run EXPLAIN QUERY PLAN (don't execute the query). plan shows SEARCH (pk or index used) -> True,
# full table SCAN -> False.
# ex: WITHOUT ROWID table with pk (metric_code, place_name, start_year, end_year):
#     filter metric_code = ? -> True, filter only on value -> False
def uses_primary_key(conn: sqlite3.Connection, sql: str, params: list) -> bool:
    pass


if __name__ == '__main__':
    # check 1: parse_year_ranges rejects "2022-2018" and "20x1" with ValueError.
    # check 2: build_query with only metric codes -> one IN clause, params count == codes count.
    # check 3: uses_primary_key is True for a metric code filter on a WITHOUT ROWID table.
    pass
