# a transaction is atomic, consistent, isolated and durable, and weak isolation allows dirty reads, phantoms and lost updates.
# the classic application bug is a lost update, where two writers read the same old balance and each writes back a value.
# fixes are an atomic update, an optimistic version column, a row lock, or a serializable retry loop.
# use sqlite3 with a file database so two connections share it, and isolation_level=None for manual begin and commit.

import sqlite3


def increment_unsafe(path: str, barrier) -> None:
    # read n, wait on the barrier so another thread reads too, then write n + 1.
    # keep it buggy on purpose so a demo can show the lost update.
    pass


def increment_optimistic(path: str, max_retries: int) -> bool:
    # add a version column and run update where id = 1 and version = ?, retrying when rowcount is 0.
    # return False if the retries are exhausted.
    pass


def insert_once(path: str, event_id: str) -> bool:
    # events has a text primary key, so insert or ignore and return True only if this call inserted.
    # check cursor.rowcount instead of selecting first, which is the idempotency key pattern.
    pass


if __name__ == '__main__':
    # run two threads of increment_unsafe with threading.Barrier(2) and assert the final n is 1.
    # hammer insert_once from many threads with one id and assert exactly one call returns True.
    pass
