# you cannot tell a crashed node from a slow one or a partitioned network; a failure detector only
# SUSPECTS. heartbeat detector: mark a node suspect when no heartbeat for timeout; too short gives false
# positives, too long gives slow failover. phi accrual outputs a suspicion level from observed intervals.
# leader election picks one coordinator. the bully algorithm: a node that notices the leader is gone
# sends ELECTION to higher ids; if none answer it declares itself leader; the highest live id wins.
# real systems (raft) use terms and majority votes to avoid two leaders (split brain).
# the simulation is synchronous: calls between toy nodes, no real network, injected clock.
# from platform: modules/uptime/ checks urls on a schedule and sends monitor.down / monitor.up events when
# state changes (see modules/uptime/README.md). that is an external prober; here you build the in-cluster version.
# explain: why does a majority quorum prevent two leaders in different network partitions?

class HeartbeatDetector:
    def __init__(self, timeout: float, now):
        pass

    def heartbeat(self, node: str) -> None:
        pass

    def suspects(self) -> set[str]:
        # nodes seen at least once whose last heartbeat is older than timeout. unseen nodes are not suspect.
        pass

    def flapping_guard(self, node: str, required_misses: int) -> bool:
        # optional hardening: require N consecutive missed checks before declaring failure.
        pass


class Cluster:
    def __init__(self, ids: list[int]):
        # alive set of node ids; leader id or None.
        pass

    def kill(self, node_id: int) -> None:
        pass

    def revive(self, node_id: int) -> None:
        pass

    def bully_election(self, starter: int) -> int:
        # starter sends ELECTION to every higher alive id; the highest alive id becomes leader.
        # return the leader id and set it. raise if the starter is dead. count messages sent.
        pass


def majority(n: int) -> int:
    # smallest vote count that two disjoint groups cannot both reach.
    pass


if __name__ == '__main__':
    # kill the leader and assert an election from any survivor picks the highest alive id.
    # with a fake clock assert a node is suspect only after timeout, and clears on the next heartbeat.
    pass
