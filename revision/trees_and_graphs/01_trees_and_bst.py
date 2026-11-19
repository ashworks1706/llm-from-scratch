# solve each node from its children's answers. bst inorder is sorted; bounds cover whole subtrees.
# recursion depth = height, O(n) when skewed, and python stops near 1000 frames.

from collections import deque


class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode | None' = None, right: 'TreeNode | None' = None):
        self.val = val
        self.left = left
        self.right = right


# 1. is_valid_bst
# True if every node's left subtree is strictly smaller and right subtree strictly larger
# (whole subtree, not just children). equal values -> False. None -> True.
# ex: 5(1, 6(3, _)) -> False (3 is right of 5),  2(1, 3) -> True,  2(2, _) -> False
# O(n): pass open bounds (low, high) down, None for missing; parent-child checks alone are wrong.
def is_valid_bst(root: TreeNode | None) -> bool:
    pass


# 2. inorder_iterative
# values in inorder (left, node, right), no recursion. None -> [].
# ex: 2(1, 3) -> [1, 2, 3]
# O(n) time, O(h) space: stack; push left spine, pop and visit, go right.
def inorder_iterative(root: TreeNode | None) -> list[int]:
    pass


# 3. lca_binary_tree
# value of the lowest node with both p and q in its subtree; a node is its own ancestor.
# values unique, tree unordered (not a bst), p != q both present. root None -> None.
# ex: 3(5(6, 2(7, 4)), 1(0, 8)): (5, 1) -> 3,  (5, 4) -> 5,  (7, 4) -> 2
# O(n) postorder: return the node when both sides report a hit.
def lca_binary_tree(root: TreeNode | None, p: int, q: int) -> int | None:
    pass


if __name__ == '__main__':
    # is_valid_bst(5(1, 6(3, _))) is False.
    # build trees from nested tuples; check inorder_iterative vs recursive inorder, incl. empty.
    pass
