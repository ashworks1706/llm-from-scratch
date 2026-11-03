# union-find tracks which items share a group, with find returning a representative and union merging two groups.
# path compression plus union by size makes each operation near O(1) amortized.
# it answers whether two nodes are connected while edges stream in, without a fresh bfs per query.
# an edge whose endpoints already share a root closes a cycle, and kruskal uses that to skip it.


class UnionFind:
    # parent and size arrays over 0 to n - 1, plus a count of groups.
    def __init__(self, n: int):
        pass

    def find(self, x: int) -> int:
        # root of x with the path compressed, iterative so a long chain cannot overflow the stack.
        pass

    def union(self, a: int, b: int) -> bool:
        # attach the smaller root under the larger and return False when the two were already joined.
        pass


def redundant_connection(edges: list[tuple[int, int]]) -> tuple[int, int]:
    # nodes are 1 to n and the edges form a tree plus one extra edge.
    # return the edge that closes the cycle, which is the first one whose union returns False.
    pass


def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> int | None:
    # (u, v, weight) are undirected edges, and the result is the minimum spanning tree weight or None if disconnected.
    # sort by weight, stop after n - 1 accepted edges, and let n = 1 give 0.
    pass


if __name__ == '__main__':
    # assert redundant_connection([(1, 2), (1, 3), (2, 3)]) == (2, 3).
    # oracle for kruskal_mst on 6 nodes is the cheapest connecting subset of n - 1 edges found by brute force.
    pass
