# claims must be atomic (select-then-update lets two workers grab one job). delivery is at least
# once, so job bodies must be idempotent. retry delays grow exponentially, capped.

import sqlite3


# 1. make_queue
# return a new in-memory connection with table jobs(id integer primary key, name text,
# status text default 'todo', attempts integer default 0).
def make_queue() -> sqlite3.Connection:
    pass


# 2. claim
# take the todo job with the smallest id: status -> 'doing', attempts += 1, return its id.
# no todo job -> None. atomic: one UPDATE with a subquery picking the job + RETURNING.
def claim(db: sqlite3.Connection) -> int | None:
    pass


# 3. finish
# write and return the new status. attempts already counts this run (claim added it).
# ok -> 'succeeded'. not ok: attempts < max_attempts -> 'todo', attempts reached max -> 'failed'.
def finish(db: sqlite3.Connection, job_id: int, ok: bool, max_attempts: int) -> str:
    pass


# 4. backoff
# attempt is 1-based. return min(base * 2 ** (attempt - 1), cap) seconds.
def backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    pass


# ---- try it ----
if __name__ == '__main__':
    # q = make_queue()
    # q.execute("insert into jobs(name) values ('a')"); q.execute("insert into jobs(name) values ('b')")
    # assert q.execute("select status, attempts from jobs where id = 1").fetchone() == ('todo', 0)
    # assert (claim(q), claim(q), claim(q)) == (1, 2, None)
    # assert q.execute("select status, attempts from jobs where id = 1").fetchone() == ('doing', 1)
    # assert finish(q, 1, False, 3) == 'todo'
    # q.execute("update jobs set attempts = 3 where id = 1")
    # assert finish(q, 1, False, 3) == 'failed'
    # assert finish(q, 2, True, 3) == 'succeeded'
    # assert abs(backoff(1) - 2.0) < 1e-6 and abs(backoff(2) - 4.0) < 1e-6 and abs(backoff(3) - 8.0) < 1e-6 and abs(backoff(6) - 60.0) < 1e-6
    pass
