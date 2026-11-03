# points are a ledger of signed rows, not a balance column. balance = SUM(points) for (user, org).
# spending is a negative row, a refund is a positive row. history is the audit trail for free.
# a bulk award is all-or-nothing: if any member key is unknown, nothing is written.
# a member with no rows sums to NULL in SQL; the API must treat that as 0 and still rank them.
# from platform: modules/points/service.py total_points(), award(), ranked_members(), leaderboard().
# from platform: modules/points/models.py Points has user_id, organization_id, points, event, timestamp.
# explain: what do you lose with a stored balance column, and what bug class do you remove?
# explain: the sum is O(rows). what index makes it fast, and when would you add a snapshot?

import sqlite3


class NotMember(Exception):
    pass


def make_db() -> sqlite3.Connection:
    # points(id integer primary key, org_id int, user_id int, points real, event text) and
    # members(org_id int, user_id int, name text). index (org_id, user_id) on points.
    pass


def balance(db: sqlite3.Connection, org_id: int, user_id: int) -> float:
    # SUM of that user's rows in that org only, 0.0 when there are none (COALESCE).
    pass


def award(db: sqlite3.Connection, org_id: int, user_ids: list[int], points: float, event: str) -> int:
    # every user must be a member of org_id else raise NotMember listing the bad ids and write nothing.
    # insert one row per user in a single transaction. return the number of rows.
    pass


def leaderboard(db: sqlite3.Connection, org_id: int, limit: int) -> list[tuple[str, float]]:
    # active members of the org with total points (0 if none), highest first, ties by name ascending.
    # LEFT JOIN so zero-point members appear. rows of other orgs must not count toward the total.
    pass


if __name__ == '__main__':
    # assert: balance of a new member is 0, award to [1, 99] with 99 unknown writes no rows,
    # a -50 row lowers the balance, the same user in two orgs has two independent balances,
    # leaderboard ties break by name.
    pass
