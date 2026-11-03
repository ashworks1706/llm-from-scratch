# multi-tenant means one table holds every org's rows, so the org id is the only wall between them.
# every lookup filters on organization_id AND the row key. an id alone is not an identity.
# a miss and a cross-tenant hit must look the same to the caller (404), or ids can be probed.
# uniqueness is per tenant too: UNIQUE(organization_id, pod_id), not UNIQUE(pod_id).
# from platform: modules/godfather/service.py _find() filters by organization_id and pod_id.
# from platform: modules/storefront/service.py _all() loads many ids for one org, 404 if any is missing.
# explain: why is "WHERE id = ?" followed by an org check in python worse than filtering in SQL?
# explain: what query would still leak even though every single-row lookup is scoped?

import sqlite3


class NotFound(Exception):
    pass


def make_db() -> sqlite3.Connection:
    # table pods(org_id int, pod_id text, name text) with UNIQUE(org_id, pod_id).
    # use an in-memory db; return the connection.
    pass


def get_pod(db: sqlite3.Connection, org_id: int, pod_id: str) -> dict:
    # one row for (org_id, pod_id) or NotFound. never query by pod_id alone.
    pass


def get_many(db: sqlite3.Connection, org_id: int, pod_ids: list[str]) -> list[dict]:
    # all-or-nothing: if any id is missing for this org, raise NotFound naming the missing ids.
    # one query, not one per id. return rows in a stable order.
    pass


def list_pods(db: sqlite3.Connection, org_id: int) -> list[dict]:
    # only this org's rows. an unfiltered SELECT here is the classic leak.
    pass


if __name__ == '__main__':
    # seed two orgs that share the SAME pod_id "p1" with different names.
    # assert org A never sees org B's p1, get_many with one foreign id raises,
    # and list_pods(A) never contains a row whose org_id is B.
    pass
