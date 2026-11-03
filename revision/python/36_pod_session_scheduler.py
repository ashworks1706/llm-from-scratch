# pods cost money while they run, so a session is a time window: start the pod a few minutes early,
# stop it when the window ends. a cron job (every 5 minutes) reconciles desired state with real state.
# group due sessions by pod: a pod stops only when no other session of it is still active.
# flags started/finished make each step happen once even if the job runs again or a call fails.
# from platform: modules/godfather/schedule.py run(), add_session() (limits: <= 24h, stop in the future), START_LEAD.
# from platform: modules/godfather/jobs.py godfather.schedule cron "*/5 * * * *".
# explain: why reconcile on every tick instead of scheduling one start and one stop event per session?
# explain: if the stop call fails, what state makes the next tick retry it?

from dataclasses import dataclass

START_LEAD = 10 * 60
MAX_LENGTH = 24 * 3600


@dataclass
class Session:
    pod_id: str
    start: int
    stop: int
    started: bool = False
    finished: bool = False


def validate(start: int, stop: int, now: int) -> None:
    # ValueError unless stop > start, stop - start <= MAX_LENGTH, and stop > now.
    pass


def tick(sessions: list[Session], running: set[str], now: int) -> dict[str, list[str]]:
    # running is the set of pod ids that are up now. mutate it as you start and stop.
    # due = not finished and start <= now + START_LEAD, grouped by pod.
    # active (stop > now) and not all started: start the pod if not running, mark those started.
    # ended (stop <= now) and nothing active and some ended one was started: stop the pod.
    # mark every ended session finished. return {"started": [...], "stopped": [...]}.
    pass


if __name__ == '__main__':
    # assert: a pod starts 10 minutes ahead, not 11; running tick twice starts once;
    # two overlapping sessions keep the pod up until the later one ends; a never-started past session
    # marks finished without stopping anything; a pod with no sessions is never touched.
    pass
