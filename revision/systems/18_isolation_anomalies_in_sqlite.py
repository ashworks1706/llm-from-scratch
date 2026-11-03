# a transaction groups statements as atomic, consistent, isolated, durable (ACID). isolation anomalies:
# dirty read, non-repeatable read, phantom, lost update, write skew. the classic app bug is a lost update:
# SELECT balance, compute in python, UPDATE balance = <value> -- two writers read the same old value.
# fixes: atomic update (UPDATE t SET n = n + 1), optimistic version column, SELECT ... FOR UPDATE
# (postgres), or a serializable retry loop. sqlite is single-writer: BEGIN IMMEDIATE takes the write lock early.
# stdlib only: sqlite3 with a file db (not :memory:) so two connections share it; isolation_level=None
# gives manual BEGIN/COMMIT control.
# from halo: packages/db/src/queries/subscription-events.ts inserts with onConflictDoNothing + returning so a
# repeated event id (the primary key) is detected in ONE atomic statement, not select-then-insert.
# explain: why does READ COMMITTED still allow a lost update on read-modify-write in application code?

import sqlite3


def setup(path: str) -> None:
    # create table counters(id INTEGER PRIMARY KEY, n INTEGER NOT NULL) and seed row id=1, n=0.
    pass


def increment_unsafe(path: str, barrier) -> None:
    # connection A: read n, wait on the barrier so another thread reads too, then write n+1.
    # this is the lost-update bug; keep it buggy so the demo can show it.
    pass


def increment_atomic(path: str) -> None:
    # one UPDATE counters SET n = n + 1 WHERE id = 1, committed. no python-side read.
    pass


def increment_optimistic(path: str, max_retries: int) -> bool:
    # add a version column; UPDATE ... WHERE id = 1 AND version = ?; if rowcount == 0 retry.
    # return False if retries are exhausted.
    pass


def insert_once(path: str, event_id: str) -> bool:
    # events(id TEXT PRIMARY KEY). INSERT OR IGNORE and return True only if this call inserted
    # (cursor.rowcount). this is the idempotency-key pattern.
    pass


if __name__ == '__main__':
    # run two threads of increment_unsafe with a threading.Barrier(2) and assert final n == 1 (lost update).
    # run 20 threads of increment_atomic and assert n == 20. hammer insert_once with one id, assert one True.
    pass
