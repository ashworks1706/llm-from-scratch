# lost update: two writers read the same old value, each writes back. fixes: atomic update,
# optimistic version column, row lock, serializable retry.
# sqlite file db (two connections share it), isolation_level=None for manual begin/commit.

import sqlite3


# 1. increment_unsafe
# buggy on purpose. counter row id 1, column n. own connection: read n, barrier.wait()
# (threading.Barrier, so all threads have read), write n + 1, commit.
def increment_unsafe(path: str, barrier) -> None:
    pass


# 2. increment_optimistic
# counter row id 1 has n and version. read n, version; then
# update set n = n + 1, version bumped, where id = 1 and version = <read version>.
# cursor.rowcount == 0 -> someone else won: re-read and retry. success -> True.
# at most max_retries attempts; all lose -> False.
def increment_optimistic(path: str, max_retries: int) -> bool:
    pass


# 3. insert_once
# events table, text primary key event id. True only if this call inserted a new row;
# existing id -> change nothing, False. use insert or ignore + cursor.rowcount, not select first
# (select-then-insert is a race). idempotency key pattern.
def insert_once(path: str, event_id: str) -> bool:
    pass


# ---- try it ----
if __name__ == '__main__':
    import os, tempfile, threading
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    setup = sqlite3.connect(path)
    setup.executescript("create table counter(id integer primary key, n integer, version integer default 0);"
                        "insert into counter(id, n) values (1, 0); create table events(id text primary key);")
    setup.commit(); setup.close()
    # barrier = threading.Barrier(2)
    # ts = [threading.Thread(target=increment_unsafe, args=(path, barrier)) for _ in range(2)]
    # [t.start() for t in ts]; [t.join() for t in ts]
    # print(sqlite3.connect(path).execute("select n from counter").fetchone())   # (1,)  lost update
    # assert increment_optimistic(path, 3) is True
    # print(insert_once(path, "e1"), insert_once(path, "e1"))                    # True False
    pass
