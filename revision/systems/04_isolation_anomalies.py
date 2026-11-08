# a transaction is atomic, consistent, isolated and durable, and weak isolation allows dirty reads, phantoms and lost updates.
# the classic application bug is a lost update, where two writers read the same old balance and each writes back a value.
# fixes are an atomic update, an optimistic version column, a row lock, or a serializable retry loop.
# use sqlite3 with a file database so two connections share it, and isolation_level=None for manual begin and commit.

import sqlite3


def increment_unsafe(path: str, barrier) -> None:
    # Problem 1: Lost Update Demo.
    # The sqlite database at path has a counter row with id 1 and a column n. Open your own
    # connection, read n, then wait on the barrier (a threading.Barrier) so that every other
    # thread has also read, and then write n + 1 and commit. This is buggy on purpose: when two
    # threads do it together, both write the same value and one increment is lost.
    #
    # Example: with n = 0 and two threads using threading.Barrier(2), both read 0, both write 1,
    # and the final n is 1 instead of 2.
    pass


def increment_optimistic(path: str, max_retries: int) -> bool:
    # Problem 2: Optimistic Increment.
    # The counter row with id 1 also has a version column. Increment n without losing updates.
    # Read n and version, then run an update that sets n to n + 1 and bumps version, with the
    # condition where id = 1 and version = the version you read. If cursor.rowcount is 0, another
    # writer got there first, so read again and retry. Return True once an update succeeds. Make at
    # most max_retries attempts and return False if all of them lose the race.
    #
    # Example: with no competing writers, the first attempt succeeds and the call returns True.
    pass


def insert_once(path: str, event_id: str) -> bool:
    # Problem 3: Idempotent Insert.
    # The database has an events table whose primary key is the text event id. Insert event_id
    # and return True only if this call inserted a new row. If the id already exists, change
    # nothing and return False. Use insert or ignore and check cursor.rowcount, not a select
    # first, because a select then insert is itself a race. This is the idempotency key pattern.
    #
    # Example: the first insert_once(path, 'e1') gives True and a second call gives False.
    pass


if __name__ == '__main__':
    # Run two threads of increment_unsafe with threading.Barrier(2) and check that the final n
    # is 1, which shows the lost update.
    # Call insert_once from many threads with the same single id and check that exactly one call
    # returns True.
    pass
