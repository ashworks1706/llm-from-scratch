# A database backed queue needs an atomic claim so two workers never run the same job. A select
# followed by an update is two statements, so both workers can see the same todo row between them.
# Delivery is at least once, so a job body must be idempotent even when the claim is perfect. Retry
# delays grow exponentially and are capped.

import sqlite3


# Problem 1: create the queue database.
#
# The function takes no arguments and returns a new in memory sqlite3 connection holding a table
# jobs with the columns id (integer primary key), name (text), status (text, default 'todo') and
# attempts (integer, default 0). A freshly inserted job therefore has status 'todo' and zero
# attempts.
#
# Example: inserting a row with only a name gives status 'todo' and attempts 0.
def make_queue() -> sqlite3.Connection:
    pass


# Problem 2: claim the oldest job.
#
# The function takes db, a connection from make_queue. It finds the todo job with the smallest id,
# sets its status to 'doing', adds one to its attempts, and returns its id. If no job has status
# 'todo' it returns None. The claim must be atomic: use one UPDATE statement with a subquery that
# picks the job and a RETURNING clause, so there is no gap between reading and writing in which a
# second worker could take the same job.
#
# Example: with jobs 1 and 2 both todo, the first claim returns 1, the second returns 2, and the
# third returns None. After the first claim job 1 has status 'doing' and attempts 1.
def claim(db: sqlite3.Connection) -> int | None:
    pass


# Problem 3: record the outcome of a job.
#
# The function takes db, job_id, an int, ok, a bool, and max_attempts, an int, and returns the job's
# new status as a string. The attempts count already includes the current run because claim added
# it. If ok is true the status becomes 'succeeded'. If ok is false and attempts is still below
# max_attempts, the status goes back to 'todo' so the job can be claimed again. If ok is false and
# attempts has reached max_attempts, the status becomes 'failed'. The new status is written to the
# database and also returned.
#
# Example: with max_attempts 3, a job on its first attempt that fails returns 'todo', and a job on
# its third attempt that fails returns 'failed'. A job that succeeds returns 'succeeded'.
def finish(db: sqlite3.Connection, job_id: int, ok: bool, max_attempts: int) -> str:
    pass


# Problem 4: exponential backoff delay.
#
# The function takes attempt, the 1-based number of the retry, base, a float that defaults to 2.0,
# and cap, a float that defaults to 60.0. It returns the number of seconds to wait before that
# retry, which is base * 2 ** (attempt - 1), but never more than cap.
#
# Example: backoff(1) gives 2.0, backoff(2) gives 4.0, backoff(3) gives 8.0, and backoff(6) gives
# 60.0 because 64 is above the cap.
def backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    pass


if __name__ == '__main__':
    # Check one: with one todo job, two claims in a row return one id and then None.
    # Check two: insert twenty jobs, start four threads that each claim until they get None, and
    # assert that no job id was claimed by more than one thread and that all twenty were claimed.
    # Each thread needs its own way to reach the same database, for example a shared file or a
    # connection created with check_same_thread=False guarded appropriately.
    pass
