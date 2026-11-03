# a scheduler picks which ready task runs next. preemption = a timer interrupt ending a quantum.
# round robin gives equal slices: fair, good response time, but small quanta add context switches.
# priority scheduling runs the highest priority first and can starve the rest; aging (raise priority
# while waiting) fixes starvation. this is a tick simulation, no real threads.
# explain: how does quantum size trade response time against switch overhead?
# explain: what is priority inversion and how does priority inheritance fix it?

from collections import deque
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    arrival: int
    burst: int
    priority: int = 0


def round_robin(tasks: list[Task], quantum: int) -> dict[str, int]:
    # return completion tick for each task name. cpu idles if nothing has arrived.
    # a task that arrives during a slice queues BEFORE the preempted task re-queues.
    # do not mutate the caller's tasks (copy remaining burst).
    pass


def priority_with_aging(tasks: list[Task], age_every: int) -> list[str]:
    # non-preemptive. at each decision pick max effective priority among arrived tasks,
    # effective = priority + waited_ticks // age_every. ties: earlier arrival, then name.
    # return the order tasks ran. assert a low priority task still finishes.
    pass


def waiting_times(completion: dict[str, int], tasks: list[Task]) -> dict[str, int]:
    # waiting = completion - arrival - burst. used to compare policies.
    pass


if __name__ == '__main__':
    # build 3-4 tasks by hand; assert exact completion ticks for quantum 2 on paper first.
    pass
