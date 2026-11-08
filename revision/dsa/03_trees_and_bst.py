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
    # Problem 1: Validate Binary Search Tree.
    # Given the root of a binary tree of integers (or None for an empty tree), return True if it
    # is a valid binary search tree and False otherwise. Valid means that for every node, all
    # values in its left subtree are strictly smaller and all values in its right subtree are
    # strictly larger. This covers the whole subtree, not just the direct children. Equal values
    # make the tree invalid, and an empty tree is valid.
    #
    # Example: a root 5 with children 1 and 6, where 6 has a left child 3, gives False because 3
    # sits in the right subtree of 5 but is smaller than 5. A root 2 with children 1 and 3 gives
    # True, and a root 2 with a left child 2 gives False.
    #
    # Aim for O(n). Pass an open interval (low, high) down the recursion, using None rather than
    # infinity for a missing bound. Comparing only parent and child is wrong.
    pass


def inorder_iterative(root: TreeNode | None) -> list[int]:
    # Problem 2: Iterative Inorder Traversal.
    # Given the root of a binary tree (or None), return the values in inorder order: the left
    # subtree, then the node, then the right subtree. Do not use recursion. An empty tree gives [].
    #
    # Example: a root 2 with children 1 and 3 gives [1, 2, 3].
    #
    # Aim for O(n) time and O(h) space for tree height h. Use a stack: push the whole left spine,
    # pop a node, visit it, then move to its right child.
    pass


def lca_binary_tree(root: TreeNode | None, p: int, q: int) -> int | None:
    # Problem 3: Lowest Common Ancestor of a Binary Tree.
    # Given the root of a binary tree whose values are unique, and two values p and q, return the
    # value of the lowest node that has both p and q in its subtree. A node counts as its own
    # ancestor, so if q is below p then the answer is p. Assume p and q both exist in the tree
    # and are different. The tree has no ordering, so it is not a search tree. If root is None,
    # return None.
    #
    # Example: take root 3 with left child 5 and right child 1, where 5 has children 6 and 2,
    # 2 has children 7 and 4, and 1 has children 0 and 8. Then lca_binary_tree(root, 5, 1) gives
    # 3, lca_binary_tree(root, 5, 4) gives 5, and lca_binary_tree(root, 7, 4) gives 2.
    #
    # Aim for O(n) with a postorder walk that returns the node when both sides report a hit.
    pass


if __name__ == '__main__':
    # Check that is_valid_bst returns False for root 5 with children 1 and 6, where 6 has a left
    # child 3.
    # Write a helper that builds trees from nested tuples, then check inorder_iterative against a
    # simple recursive inorder on several trees, including an empty one.
    pass
