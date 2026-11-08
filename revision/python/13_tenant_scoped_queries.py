# In a multi tenant table the tenant id is the only wall between customers. Every lookup filters on
# the tenant id and the row key together in SQL, never on the row key alone. A missing row and
# another tenant's row must look identical to the caller. Uniqueness is per tenant too, so the
# constraint covers both columns.

import sqlite3


# Raised when the requested pod does not exist for the requesting org.
class NotFound(Exception):
    pass


# Problem 1: create the pods database.
#
# The function takes no arguments and returns a new in memory sqlite3 connection holding one table
# named pods with the columns org_id (integer), pod_id (text) and name (text), and a UNIQUE
# constraint over the pair (org_id, pod_id). Two orgs may therefore both own a pod called "p1", but
# one org cannot have it twice.
#
# Example: inserting (1, "p1", "alpha") and (2, "p1", "beta") succeeds, and inserting (1, "p1", "x")
# again raises sqlite3.IntegrityError.
def make_db() -> sqlite3.Connection:
    pass


# Problem 2: fetch one pod for an org.
#
# The function takes db, a connection from make_db, org_id, an int, and pod_id, a string. It returns
# the single matching row as a dict with the keys "org_id", "pod_id" and "name". The query filters on
# both org_id and pod_id with bound parameters. If no row matches, including the case where the
# pod_id exists only under a different org, it raises NotFound with the same message either way, so
# the caller cannot tell the two cases apart.
#
# Example: with (1, "p1", "alpha") and (2, "p1", "beta") stored, get_pod(db, 1, "p1") gives
# {"org_id": 1, "pod_id": "p1", "name": "alpha"}, and get_pod(db, 1, "zzz") raises NotFound.
def get_pod(db: sqlite3.Connection, org_id: int, pod_id: str) -> dict:
    pass


# Problem 3: fetch several pods for an org.
#
# The function takes db, org_id, an int, and pod_ids, a list of strings. It returns a list of row
# dicts, one per requested id, ordered by pod_id so the order is stable and does not depend on the
# order of the request. The call is all or nothing: if any requested id is missing for this org, it
# raises NotFound and the message names every missing id. An empty pod_ids list gives an empty list.
# Duplicate ids in the request count once. Use a single query with one bound placeholder per id, not
# one query per id.
#
# Example: with org 1 owning "p1" and "p2", get_many(db, 1, ["p2", "p1"]) gives the rows for "p1"
# then "p2". get_many(db, 1, ["p1", "p9"]) raises NotFound naming "p9".
def get_many(db: sqlite3.Connection, org_id: int, pod_ids: list[str]) -> list[dict]:
    pass


if __name__ == '__main__':
    # Seed two orgs, a and b, that both own the pod id "p1" with different names.
    # Check one: get_pod for org a returns org a's name and never org b's.
    # Check two: get_many for org a with an id that only org b owns raises NotFound.
    pass
