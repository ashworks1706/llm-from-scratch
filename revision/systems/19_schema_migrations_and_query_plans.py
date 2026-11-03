# a migration is an ordered, versioned, applied-once schema change. keep a schema_migrations table of
# applied ids; apply pending ones in order inside a transaction; never edit an applied migration.
# an index trades write cost and space for read speed. EXPLAIN QUERY PLAN shows SCAN (full table) vs
# SEARCH ... USING INDEX. composite index column order matters: (a, b) serves a, and a+b, not b alone.
# a covering index answers from the index without touching the table.
# stdlib only: sqlite3.
# from platform: AGENTS.md says nothing creates tables at startup, alembic/ holds migrations, and
# make ci runs alembic check. from halo: packages/db/src/migrations/ holds timestamp-named dirs.
# explain: why can adding an index on a hot postgres table take down writes, and what does CONCURRENTLY do?

import sqlite3


def apply_migrations(conn: sqlite3.Connection, migrations: list[tuple[str, str]]) -> list[str]:
    # migrations = [(id, sql)]. ensure schema_migrations(id TEXT PRIMARY KEY, applied_at TEXT) exists.
    # apply unapplied ones in sorted id order, each in its own transaction; a failure rolls back that one
    # and stops. return the ids applied by THIS call. running twice applies nothing the second time.
    # detect a duplicate id in the input list with ValueError.
    pass


def plan(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[str]:
    # return the "detail" column of EXPLAIN QUERY PLAN rows as a list of strings.
    pass


def uses_index(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> bool:
    # true when the plan contains SEARCH ... USING (COVERING) INDEX and no full SCAN of the main table.
    pass


if __name__ == '__main__':
    # build orders(id, user_id, status, created_at) with 10k rows. assert the query for one user_id
    # scans before CREATE INDEX and searches after. show an index on (status, user_id) does NOT help
    # "WHERE user_id = ?" and an index on (user_id, created_at) helps "WHERE user_id=? ORDER BY created_at".
    pass
