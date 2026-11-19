# claims must be atomic (select-then-update lets two workers grab one job). delivery is at least
# once, so job bodies must be idempotent. retry delays grow exponentially, capped.

import sqlite3


# 1. make_queue
# return a new in-memory connection with table jobs(id integer primary key, name text,
# status text default 'todo', attempts integer default 0).
# ex: insert with only a name -> status 'todo', attempts 0
def make_queue() -> sqlite3.Connection:
    pass


# 2. claim
# take the todo job with the smallest id: status -> 'doing', attempts += 1, return its id.
# no todo job -> None. atomic: one UPDATE with a subquery picking the job + RETURNING.
# ex: jobs 1, 2 todo -> claims return 1, 2, None. after the first, job 1 is 'doing', attempts 1
def claim(db: sqlite3.Connection) -> int | None:
    pass


# 3. finish
# write and return the new status. attempts already counts this run (claim added it).
# ok -> 'succeeded'. not ok: attempts < max_attempts -> 'todo', attempts reached max -> 'failed'.
# ex: max_attempts 3: fail on 1st attempt -> 'todo', fail on 3rd -> 'failed', success -> 'succeeded'
def finish(db: sqlite3.Connection, job_id: int, ok: bool, max_attempts: int) -> str:
    pass


# 4. backoff
# attempt is 1-based. return min(base * 2 ** (attempt - 1), cap) seconds.
# ex: backoff(1) -> 2.0, (2) -> 4.0, (3) -> 8.0, (6) -> 60.0 (64 capped)
def backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    pass


if __name__ == '__main__':
    # check 1: one todo job -> two claims return its id, then None.
    # check 2: 20 jobs, 4 threads each claiming until None -> no id claimed twice, all 20 claimed.
    # each thread needs its own way to the same db (shared file, or check_same_thread=False + guard).
    pass
