# a db-backed queue needs an atomic claim: two workers must never run the same job.
# on postgres this is SELECT ... FOR UPDATE SKIP LOCKED (workers skip rows another worker holds).
# sqlite has no row locks, so the toy uses one conditional UPDATE and checks the rowcount.
# delivery is at-least-once: a crash after the work but before the ack runs the job again.
# from platform: core/jobs.py job(retry=N) hands retries to Procrastinate; platform only calls task.defer().
# from platform: core/jobs.py _execute() re-raises so the queue can retry; the inline backend (_run) swallows.
# the SKIP LOCKED claim and backoff live inside Procrastinate, not in this repo. this toy is the idea.
# explain: why is "SELECT todo, then UPDATE doing" in two statements a double-run bug?
# explain: why must a job body be idempotent even with a perfect claim?

import sqlite3


def make_queue() -> sqlite3.Connection:
    # table jobs(id integer primary key, name text, status text default 'todo', attempts int default 0).
    pass


def enqueue(db: sqlite3.Connection, name: str) -> int:
    # insert a todo job, return its id.
    pass


def claim(db: sqlite3.Connection) -> int | None:
    # atomically move the oldest todo job to doing and bump attempts. return its id, or None.
    # one statement (UPDATE ... WHERE id = (SELECT ... LIMIT 1) RETURNING id). no read-then-write gap.
    pass


def finish(db: sqlite3.Connection, job_id: int, ok: bool, max_attempts: int) -> str:
    # ok -> 'succeeded'. failed with attempts < max_attempts -> back to 'todo' (retry).
    # failed at max_attempts -> 'failed'. return the new status.
    pass


def backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    # delay before retry number attempt (1-based): base * 2**(attempt-1), never above cap.
    pass


if __name__ == '__main__':
    # assert: two claim() calls on one job give one id and one None,
    # a job failing twice with max_attempts=3 is todo again, the third failure marks it failed,
    # backoff(1..10) is non decreasing and capped. run 4 threads claiming 20 jobs: no id twice.
    pass
