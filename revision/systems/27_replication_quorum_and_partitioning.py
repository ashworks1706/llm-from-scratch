# leader-follower replication: writes go to the leader, which appends to a log and ships entries to
# followers in order; async replication is fast but followers lag and a failover can lose acked writes;
# sync replication waits and trades latency for durability. quorum systems with N replicas, W write acks
# and R read acks guarantee the read set overlaps the latest write set when R + W > N.
# partitioning: hash partitioning spreads load evenly but loses range scans; range partitioning keeps
# ranges (and hot spots on monotonic keys like timestamps). resharding is the real cost (see 16_ ring).
# explain: what anomaly can a client see reading from a lagging follower, and what is read-your-writes?
# explain: with N=3, W=2, R=2, how can a read still return stale data? (sloppy quorum, concurrent writes)

import bisect


class Replica:
    def __init__(self, name: str):
        # log: list of (seq, key, value); store: dict; applied: highest applied seq.
        pass

    def apply(self, seq: int, key: str, value: str) -> bool:
        # apply in order only: accept seq == applied + 1; ignore a duplicate (<= applied);
        # return False for a gap (caller must resend the missing entries).
        pass


class Leader(Replica):
    def __init__(self, name: str, followers: list[Replica]):
        pass

    def write(self, key: str, value: str, sync_acks: int) -> int:
        # append locally, ship to followers, return seq. only `sync_acks` followers must apply before
        # returning; the rest catch up later (call catch_up).
        pass

    def catch_up(self, follower: Replica) -> int:
        # send missing entries starting after follower.applied. return how many were sent.
        pass


class QuorumStore:
    def __init__(self, n: int, w: int, r: int):
        # validate 1 <= w, r <= n. versions are (counter) per key per replica; replicas can be marked down.
        pass

    def put(self, key: str, value: str, down: set[int]) -> bool:
        # write to all UP replicas; succeed only if at least w acked.
        pass

    def get(self, key: str, down: set[int]) -> str | None:
        # read from r up replicas (raise if fewer than r are up); return the highest-version value.
        pass


def range_partition(boundaries: list[int], key: int) -> int:
    # boundaries sorted ascending; partition i holds keys < boundaries[i]; last holds the rest. bisect.
    pass
