# round robin spreads requests evenly when work is uniform. least connections adapts when requests have
# very different durations. hash by key gives stickiness (cache locality, sessions), but plain hash % N
# remaps almost every key when N changes. a consistent hash ring puts servers on a circle; a key goes to
# the next server clockwise, so adding or removing one server moves only about 1/N of the keys.
# virtual nodes (many points per server) even out the load.
# stdlib only: bisect, hashlib. use a stable hash (md5/sha1), NOT python's hash() (salted per process).
# explain: why does least-connections need the load balancer to see connection close events?
# explain: how would you give a bigger server more virtual nodes?

import bisect
import hashlib


class RoundRobin:
    def __init__(self, servers: list[str]):
        pass

    def pick(self) -> str:
        # cycle in order; raise a clear error if there are no servers.
        pass


class LeastConnections:
    def __init__(self, servers: list[str]):
        pass

    def acquire(self) -> str:
        # pick the server with the fewest active connections (ties: first in list), count it.
        pass

    def release(self, server: str) -> None:
        # never go below 0; unknown server -> ValueError.
        pass


class HashRing:
    def __init__(self, vnodes: int = 100):
        pass

    def add(self, server: str) -> None:
        pass

    def remove(self, server: str) -> None:
        pass

    def lookup(self, key: str) -> str:
        # bisect on the sorted ring positions, wrap around past the end. O(log n).
        pass


if __name__ == '__main__':
    # 10000 keys on 5 servers: assert max load within ~35% of the mean. add a 6th server and assert
    # moved keys are about 1/6, and every moved key now maps to the new server (compare with hash % N).
    pass
