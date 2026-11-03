# a tree is a recursive shape: solve the node from the answers of its children.
# dfs order: preorder (node first), inorder (left, node, right), postorder (node last). inorder of a BST is sorted.
# bfs by level uses a deque and the count of nodes currently in the queue.
# BST invariant: every node in the left subtree < node < every node in the right subtree, not just the children.
# recursion depth equals tree height: O(log n) balanced, O(n) skewed, and python's default limit is about 1000.
# explain: when would you write inorder iteratively, and what does the explicit stack hold?
# explain: LCA in a BST uses order; LCA in a plain tree does not. what extra cost does that add?

from collections import deque


class TreeNode:
    def __init__(self, val: int = 0, left: 'TreeNode | None' = None, right: 'TreeNode | None' = None):
        self.val = val
        self.left = left
        self.right = right


def inorder_iterative(root: TreeNode | None) -> list[int]:
    # no recursion. push left spine, pop, visit, move right. O(n) time, O(h) space. None gives [].
    pass


def level_order(root: TreeNode | None) -> list[list[int]]:
    # values grouped by depth. snapshot len(queue) at the start of each level. O(n).
    pass


def max_depth_and_diameter(root: TreeNode | None) -> tuple[int, int]:
    # (height in nodes, diameter in edges). one postorder pass: at each node the best path through it is left + right height.
    # empty tree is (0, 0). a single node is (1, 0).
    pass


def is_valid_bst(root: TreeNode | None) -> bool:
    # pass an open interval (low, high) down; checking only parent versus child is wrong.
    # equal values are invalid. use None, not float infinity, for open bounds. O(n).
    pass


def lca_bst(root: TreeNode, p: int, q: int) -> int:
    # lowest common ancestor in a BST; both values exist. walk down: both smaller go left, both larger go right,
    # otherwise this node splits them. O(h), O(1) space. a node can be its own ancestor.
    pass


def lca_binary_tree(root: TreeNode | None, p: int, q: int) -> int | None:
    # no ordering. postorder: return the node if it matches, else combine children; both sides non-null means this node.
    # O(n) time. assume p and q exist and are distinct.
    pass


if __name__ == '__main__':
    # build trees from nested tuples with a helper, assert level_order on a 3 level tree,
    # assert is_valid_bst rejects root 5 with children 1 and 6 where 6 has a left child 3: 3 sits right of 5 but is smaller.
    pass
