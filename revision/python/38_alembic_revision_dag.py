# alembic migrations form a DAG. each revision names its down_revision: None for the root, a string,
# or a tuple for a merge revision. the heads are revisions nobody points at.
# two heads means two branches were merged into main separately; upgrade head is ambiguous until a merge revision joins them.
# upgrade order is a topological order from the root. a merge revision runs after all its parents.
# from platform: alembic/versions/d2ba7436c1b7_initial_schema.py is the root (down_revision = None).
# from platform: alembic/versions/19e8198d6e81_merge_token_limits_and_webhooks.py has down_revision = ("010e740878f3", "a1358e0da9ef").
# from platform: make ci runs alembic check, which fails when models and migrations drift.
# explain: how do two feature branches each adding a migration produce two heads, and how do you fix it?
# explain: why must tests run migrations from scratch on an empty db, not only create_all?


def normalize(down: str | tuple[str, ...] | None) -> tuple[str, ...]:
    # None -> (), "a" -> ("a",), tuple stays a tuple.
    pass


def heads(revs: dict[str, str | tuple[str, ...] | None]) -> list[str]:
    # revision ids that no other revision lists as a parent, sorted. unknown parent id -> ValueError.
    pass


def upgrade_order(revs: dict[str, str | tuple[str, ...] | None]) -> list[str]:
    # parents always before children, deterministic (tie-break by id). cycle -> ValueError.
    # Kahn's algorithm: O(V + E).
    pass


def path_to(revs: dict[str, str | tuple[str, ...] | None], target: str) -> list[str]:
    # the revisions needed to reach target: target and all its ancestors, in upgrade order.
    pass


if __name__ == '__main__':
    # build a chain a<-b<-c, branch d off b, and merge (c, d) into e.
    # assert: heads before the merge are [c, d], after it [e]; e appears after both c and d;
    # a cycle raises; path_to(c) excludes d.
    pass
