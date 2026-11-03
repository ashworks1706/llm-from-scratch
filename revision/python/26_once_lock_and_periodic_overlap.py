# a job made with once=("org_id",) is unique per those argument values: defer() while the same
# key is queued or running raises JobBusy (409), so one org cannot flood the queue.
# the lock must be released in a finally, or one crash locks that key until restart.
# periodic jobs follow a schedule; a run that is due while the last run still lives is skipped.
# the next due time comes from now, not from the old due time, so a long stall does not cause a burst.
# from platform: core/jobs.py _lock_key(), defer(), _run_in_slot() (slot semaphore + lock release).
# from platform: core/jobs.py _start_due() decides which periodic jobs start, using croniter.
# explain: why check-and-add of the lock key under one mutex and not two separate steps?
# explain: what happens to a periodic job whose run takes longer than its interval?

import threading


class JobBusy(Exception):
    pass


def lock_key(name: str, once: tuple[str, ...] | None, kwargs: dict) -> str | None:
    # None when once is None. else name + ":" + "arg=value" joined by ":" in the order of once.
    # once=() means one run at a time for the whole job.
    pass


class InlineQueue:
    def __init__(self):
        self.running: set[str] = set()
        self.mutex = threading.Lock()

    def acquire(self, key: str | None) -> None:
        # None -> no lock. key already held -> JobBusy. else hold it. one critical section.
        pass

    def release(self, key: str | None) -> None:
        # drop the key if held. safe to call twice.
        pass


def start_due(jobs: dict[str, float], due: dict[str, float], alive: set[str], now: float) -> list[str]:
    # jobs maps name -> interval seconds. for each job with due[name] <= now:
    # set due[name] = now + interval, and start it only if it is not in alive. return started names.
    pass


if __name__ == '__main__':
    # assert: same key twice raises JobBusy, different org ids do not, release then acquire works,
    # a still-alive job is skipped but its due time still moves, and a job not yet due does not start.
    pass
