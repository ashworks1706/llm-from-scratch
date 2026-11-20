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
# O(n): pass open bounds (low, high) down, None for missing; parent-child checks alone are wrong.
def is_valid_bst(root: TreeNode | None) -> bool:
    pass


# 2. inorder_iterative
# values in inorder (left, node, right), no recursion. None -> [].
# O(n) time, O(h) space: stack; push left spine, pop and visit, go right.
def inorder_iterative(root: TreeNode | None) -> list[int]:
    pass


# 3. lca_binary_tree
# value of the lowest node with both p and q in its subtree; a node is its own ancestor.
# values unique, tree unordered (not a bst), p != q both present. root None -> None.
# O(n) postorder: return the node when both sides report a hit.
def lca_binary_tree(root: TreeNode | None, p: int, q: int) -> int | None:
    pass


# ---- try it ----
if __name__ == '__main__':
    T = TreeNode
    bad = T(5, T(1), T(6, T(3), None))
    good = T(2, T(1), T(3))
    # assert (is_valid_bst(bad), is_valid_bst(good)) == (False, True)
    # assert is_valid_bst(T(2, T(2), None)) is False
    # assert inorder_iterative(good) == [1, 2, 3]
    tree = T(3, T(5, T(6), T(2, T(7), T(4))), T(1, T(0), T(8)))
    # assert (lca_binary_tree(tree, 5, 1), lca_binary_tree(tree, 5, 4), lca_binary_tree(tree, 7, 4)) == (3, 5, 2)
    pass
