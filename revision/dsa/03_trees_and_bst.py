# a tree is a recursive shape, so solve each node from the answers of its children.
# the inorder walk of a bst is sorted, and bfs by level snapshots the queue length at each level.
# the bst invariant covers the whole subtree, so every left descendant is smaller than the node, not just the child.
# recursion depth equals tree height, which is O(n) for a skewed tree and python stops near 1000 frames.

from collections import deque


class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode | None' = None, right: 'TreeNode | None' = None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: TreeNode | None) -> bool:
    # pass an open interval down the recursion, since comparing only parent and child is wrong.
    # equal values are invalid and the bounds are None rather than infinity, in O(n).
    pass


def inorder_iterative(root: TreeNode | None) -> list[int]:
    # no recursion, so push the left spine, pop, visit and move right, in O(n) time and O(h) space.
    # an empty tree gives [].
    pass


def lca_binary_tree(root: TreeNode | None, p: int, q: int) -> int | None:
    # no ordering is available, so use postorder and return the node that has both values below it.
    # assume p and q exist and are distinct, and a node may be its own ancestor.
    pass


if __name__ == '__main__':
    # assert is_valid_bst rejects root 5 with children 1 and 6 where 6 has a left child 3.
    # build trees from nested tuples with a helper and assert inorder_iterative against a recursive version.
    pass
