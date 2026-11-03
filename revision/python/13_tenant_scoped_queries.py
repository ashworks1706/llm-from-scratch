# in a multi-tenant table the tenant id is the only wall between customers.
# every lookup filters on the tenant id and the row key in sql, never the row key alone.
# a missing row and another tenant's row must look identical to the caller.
# uniqueness is per tenant too, so the constraint covers both columns.

import sqlite3


class NotFound(Exception):
    pass


def make_db() -> sqlite3.Connection:
    # in-memory table pods(org_id int, pod_id text, name text) with UNIQUE(org_id, pod_id).
    pass


def get_pod(db: sqlite3.Connection, org_id: int, pod_id: str) -> dict:
    # return the one row for (org_id, pod_id) or raise NotFound.
    pass


def get_many(db: sqlite3.Connection, org_id: int, pod_ids: list[str]) -> list[dict]:
    # all or nothing, so raise NotFound naming the ids missing for this org.
    # use one query, not one per id, and return rows in a stable order.
    pass


if __name__ == '__main__':
    # seed two orgs that share the pod id "p1" with different names.
    # assert that org a never sees org b's row and that get_many with a foreign id raises.
    pass
