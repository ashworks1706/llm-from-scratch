# a database-backed queue needs an atomic claim so two workers never run the same job.
# a select followed by an update is two statements, so both workers can see the same todo row.
# delivery is at least once, so a job body must be idempotent even with a perfect claim.
# retry delays grow exponentially and are capped.

import sqlite3


def make_queue() -> sqlite3.Connection:
    # table jobs(id integer primary key, name text, status text default 'todo', attempts int default 0).
    pass


def claim(db: sqlite3.Connection) -> int | None:
    # atomically move the oldest todo job to doing, bump attempts, and return its id or None.
    # use one UPDATE with a subquery and RETURNING, with no read-then-write gap.
    pass


def finish(db: sqlite3.Connection, job_id: int, ok: bool, max_attempts: int) -> str:
    # success gives 'succeeded', a failure below max_attempts goes back to 'todo'.
    # a failure at max_attempts gives 'failed'. return the new status.
    pass


def backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    # delay before the 1-based retry number attempt is base * 2**(attempt - 1), never above cap.
    pass


if __name__ == '__main__':
    # assert that two claims on one job give one id and one None.
    # run four threads claiming twenty jobs and assert no id is claimed twice.
    pass
