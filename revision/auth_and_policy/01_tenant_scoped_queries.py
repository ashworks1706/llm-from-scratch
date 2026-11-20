# every lookup filters tenant id + row key together in SQL. a missing row and another tenant's row
# look identical to the caller. uniqueness is per tenant: (org_id, pod_id).

import sqlite3


# raised when the requested pod does not exist for the requesting org.
class NotFound(Exception):
    pass


# 1. make_db
# return a new in-memory connection with table pods(org_id integer, pod_id text, name text)
# and UNIQUE(org_id, pod_id).
def make_db() -> sqlite3.Connection:
    pass


# 2. get_pod
# return the matching row as {"org_id", "pod_id", "name"}; filter on both columns, bound params.
# no match (incl. pod_id existing only under another org) -> NotFound, same message either way.
def get_pod(db: sqlite3.Connection, org_id: int, pod_id: str) -> dict:
    pass


# 3. get_many
# return row dicts, one per requested id, ordered by pod_id (independent of request order).
# all or nothing: any id missing for this org -> NotFound whose message names every missing id.
# [] -> []. duplicate ids count once. one query with one bound placeholder per id.
def get_many(db: sqlite3.Connection, org_id: int, pod_ids: list[str]) -> list[dict]:
    pass


# ---- try it ----
if __name__ == '__main__':
    # db = make_db()
    # db.executemany("insert into pods values (?, ?, ?)", [(1, "p1", "alpha"), (2, "p1", "beta"), (1, "p2", "gamma")])
    # db.execute("insert into pods values (1, 'p1', 'x')")      # raises sqlite3.IntegrityError
    # print(get_pod(db, 1, "p1"))                   # {'org_id': 1, 'pod_id': 'p1', 'name': 'alpha'}
    # print(get_many(db, 1, ["p2", "p1"]))          # rows p1 then p2 (org 1)
    # get_pod(db, 1, "zzz")                         # raises NotFound
    # get_many(db, 1, ["p1", "p9"])                 # raises NotFound naming p9
    pass
