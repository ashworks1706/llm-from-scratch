# lost update: two writers read the same old value, each writes back. fixes: atomic update,
# optimistic version column, row lock, serializable retry.
# sqlite file db (two connections share it), isolation_level=None for manual begin/commit.

import sqlite3


# 1. increment_unsafe
# buggy on purpose. counter row id 1, column n. own connection: read n, barrier.wait()
# (threading.Barrier, so all threads have read), write n + 1, commit.
# ex: n = 0, two threads, Barrier(2) -> both read 0, both write 1, final n = 1 (not 2)
def increment_unsafe(path: str, barrier) -> None:
    pass


# 2. increment_optimistic
# counter row id 1 has n and version. read n, version; then
# update set n = n + 1, version bumped, where id = 1 and version = <read version>.
# cursor.rowcount == 0 -> someone else won: re-read and retry. success -> True.
# at most max_retries attempts; all lose -> False.
# ex: no competing writers -> first attempt succeeds -> True
def increment_optimistic(path: str, max_retries: int) -> bool:
    pass


# 3. insert_once
# events table, text primary key event id. True only if this call inserted a new row;
# existing id -> change nothing, False. use insert or ignore + cursor.rowcount, not select first
# (select-then-insert is a race). idempotency key pattern.
# ex: insert_once(path, 'e1') -> True, again -> False
def insert_once(path: str, event_id: str) -> bool:
    pass


if __name__ == '__main__':
    # two increment_unsafe threads with Barrier(2) -> final n is 1 (lost update).
    # insert_once from many threads with one id -> exactly one True.
    pass
