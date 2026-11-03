# a b-tree keeps keys sorted in wide nodes so one disk page holds many keys: height stays tiny
# (log base ~100 of n), so a lookup is a few page reads. order-3 here means at most 3 children and 2 keys
# per node (a "2-3 tree"). insert goes to a leaf; an overfull node (3 keys) splits at the middle and pushes
# the middle key up; splitting the root grows the tree by one level, so ALL leaves stay at the same depth.
# b+trees keep values only in leaves linked together for range scans; databases mostly use that variant.
# explain: why do databases prefer b-trees over binary trees on disk, and what do page splits cost writes?
# explain: how does an LSM tree (see 22_) trade read cost for write throughput against this?

from dataclasses import dataclass, field


@dataclass
class Node:
    keys: list = field(default_factory=list)
    children: list = field(default_factory=list)  # empty for a leaf

    def is_leaf(self) -> bool:
        pass


class BTree:
    def __init__(self):
        pass

    def search(self, key) -> bool:
        # descend by comparing against the node keys (bisect). O(height).
        pass

    def insert(self, key) -> None:
        # duplicate keys are ignored. recursion may return (promoted_key, right_node) on a split.
        # if the root splits, make a new root with the single promoted key.
        pass

    def inorder(self) -> list:
        # must be sorted and complete; use it to cross-check against sorted(set(keys)).
        pass

    def height(self) -> int:
        # number of levels; assert every leaf is at this depth in tests.
        pass


if __name__ == '__main__':
    # insert 1..200 in random order several times: assert inorder == sorted unique keys, all leaves
    # equal depth, and each node has 1 or 2 keys. assert search of a missing key is False.
    pass
