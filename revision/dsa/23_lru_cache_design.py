# lru cache: get and put both in O(1). a dict gives O(1) lookup; a doubly linked list gives O(1) move-to-front and evict-last.
# the dict maps key -> node, the list orders nodes by recency. head side is most recent, tail side is next to evict.
# sentinel head and tail nodes remove every empty-list and one-node special case.
# python shortcut: collections.OrderedDict has move_to_end and popitem(last=False). write the manual version first.
# from platform: core/cache.py TTLCache uses an OrderedDict, move_to_end on set, and popitem(last=False) past max_entries.
# its get does not move the key, so eviction there follows write order, not read order. an LRU would move on get.
# from piramid: apps/engine/model/src/embeddings/cache.rs wraps an Embedder with a Mutex around an LruCache keyed by text.
# prefix caches evict by recency the same way: see inference/systems/34_prefix_caching.rs.
# explain: why does the cache need a lock in a threaded server, and why is one lock around get and put the simple answer?
# explain: what does LFU need that LRU does not, and why is O(1) LFU harder?

from collections import OrderedDict


class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        # capacity >= 1. dict key -> Node plus two sentinels linked head <-> tail.
        pass

    def get(self, key: int) -> int:
        # value or -1. a hit moves the node to the front. O(1).
        pass

    def put(self, key: int, value: int) -> None:
        # update in place and move to front if present; otherwise insert at front and evict the tail's prev when over capacity.
        # evicting must also delete the dict entry: the node stores its key for that reason.
        pass


class OrderedDictLRU:
    # same contract with OrderedDict: move_to_end(key) on hit, popitem(last=False) on overflow.
    def __init__(self, capacity: int):
        pass


class TTLLRU:
    # extension: each entry also expires after ttl seconds, using an injected clock function so tests are deterministic.
    # an expired entry counts as a miss and is removed on access. capacity eviction still follows recency.
    def __init__(self, capacity: int, ttl: float, clock):
        pass


if __name__ == '__main__':
    # replay capacity 2: put 1, put 2, get 1, put 3 (evicts 2), get 2 -> -1, put 4 (evicts 1), get 1 -> -1, get 3, get 4.
    # run the same random op sequence on LRUCache and OrderedDictLRU and assert identical results.
    pass
