# a fact table keyed by (metric, place, start_year, end_year) answers point and period lookups through its composite primary key.
# the leftmost key columns must match the equality filters used most often or the index cannot be used.
# user supplied year ranges are validated at the boundary and rejected with a clear error instead of returning empty data.
# every filter value travels as a bound parameter so one query shape is reused for any selection.

import re
import sqlite3

YEAR_RANGE = re.compile(r"^(\d{4})(?:-(\d{4}))?$")


def parse_year_ranges(values: list[str] | None) -> list[tuple[int, int]]:
    # "2020" means (2020, 2020) and "2018-2022" means (2018, 2022).
    # raise ValueError for malformed text or when end precedes start, and return [] for None or an empty list.
    pass


def build_query(metric_codes: list[str], place_names: list[str], ranges: list[tuple[int, int]]) -> tuple[str, list]:
    # return (sql, params) selecting matching rows, leaving out any clause whose input list is empty.
    # ranges become an or of (start_year = ? and end_year = ?) pairs, with one placeholder per value in each in list.
    pass


def uses_primary_key(conn: sqlite3.Connection, sql: str, params: list) -> bool:
    # return True when explain query plan reports a search on the primary key or an index rather than a full scan.
    pass


if __name__ == '__main__':
    # assert parse_year_ranges rejects "2022-2018" and "20x1".
    # assert build_query with only metric codes produces one in clause and the matching param count.
    # assert uses_primary_key is True for a metric code filter on a without rowid table.
    pass
