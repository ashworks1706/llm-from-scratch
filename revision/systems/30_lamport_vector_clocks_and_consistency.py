# physical clocks drift and cannot order events across machines. lamport clock: a counter; local event
# or send increments it; receive sets it to max(local, msg) + 1. if a happened-before b then L(a) < L(b),
# but not the converse. vector clocks keep one counter per node; compare entrywise: a < b if every entry
# <= and one is <; if neither dominates, the events are CONCURRENT (a conflict to resolve).
# consistency models, strongest to weakest: linearizable (one real-time order), sequential, causal,
# read-your-writes / monotonic reads, eventual. CAP: during a partition choose consistency or availability.
# explain: give an example where a stale read is acceptable and one where it is a bug.
# explain: why do last-write-wins registers lose data, and what do CRDTs do instead?

class Lamport:
    def __init__(self):
        pass

    def tick(self) -> int:
        # local event.
        pass

    def send(self) -> int:
        # timestamp to attach to an outgoing message.
        pass

    def receive(self, ts: int) -> int:
        pass


class VectorClock:
    def __init__(self, node: str, nodes: list[str]):
        pass

    def tick(self) -> dict[str, int]:
        # increment own entry; return a copy.
        pass

    def merge(self, other: dict[str, int]) -> dict[str, int]:
        # receive: entrywise max, then tick own entry. return a copy.
        pass


def compare(a: dict[str, int], b: dict[str, int]) -> str:
    # "before", "after", "equal" or "concurrent". missing keys count as 0.
    pass


def merge_siblings(versions: list[tuple[dict[str, int], str]]) -> list[tuple[dict[str, int], str]]:
    # drop every version dominated by another; keep the concurrent ones (siblings) for the app to resolve.
    pass


if __name__ == '__main__':
    # replay a 3-node message diagram by hand, assert exact vector values at each step,
    # and assert two writes made on different nodes without communication compare as concurrent.
    pass
