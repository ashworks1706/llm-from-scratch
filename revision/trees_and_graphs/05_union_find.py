# union-find: find returns a group's root, union merges groups. path compression + union by size
# make both near O(1) amortized. an edge whose endpoints share a root closes a cycle.


# 1. UnionFind
# items 0..n - 1, each starting in its own group. keep parent, size and a group count.
# find(x): root of x's group, with path compression, iterative (no stack overflow).
# union(a, b): attach smaller root under larger, update size and count.
#   True if merged, False if already same group.
class UnionFind:
    def __init__(self, n: int):
        pass

    def find(self, x: int) -> int:
        pass

    def union(self, a: int, b: int) -> bool:
        pass


# 2. redundant_connection
# undirected edges on nodes 1..n forming a tree plus one extra edge.
# return the first edge (in order) whose endpoints are already connected (union returns False).
def redundant_connection(edges: list[tuple[int, int]]) -> tuple[int, int]:
    pass


# 3. kruskal_mst
# nodes 0..n - 1, undirected (u, v, weight). return total mst weight.
# disconnected -> None; n == 1 -> 0.
# sort by weight, accept when union is True, stop after n - 1 accepted.
def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> int | None:
    pass


# ---- try it ----
if __name__ == '__main__':
    # uf = UnionFind(3)
    # print(uf.union(0, 1), uf.union(1, 0), uf.count)   # True False 2
    # assert redundant_connection([(1, 2), (1, 3), (2, 3)]) == (2, 3)
    # assert redundant_connection([(1, 2), (2, 3), (3, 4), (1, 4), (1, 5)]) == (1, 4)
    # assert kruskal_mst(4, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 10), (0, 2, 5)]) == 6
    # assert kruskal_mst(4, [(0, 1, 1), (2, 3, 1)]) is None
    pass
