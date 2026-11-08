# union-find tracks which items share a group, with find returning a representative and union merging two groups.
# path compression plus union by size makes each operation near O(1) amortized.
# it answers whether two nodes are connected while edges stream in, without a fresh bfs per query.
# an edge whose endpoints already share a root closes a cycle, and kruskal uses that to skip it.


class UnionFind:
    # Problem 1: Union-Find Structure.
    # Build a structure over the items 0 to n - 1, starting with each item in its own group. It
    # keeps a parent array, a size array and a count of how many groups exist. Both operations
    # below must be near O(1) amortized.
    def __init__(self, n: int):
        pass

    def find(self, x: int) -> int:
        # Return the representative (root) of the group that contains x. Compress the path so
        # later finds are faster. Write it iteratively so a long chain cannot overflow the stack.
        # Two items are in the same group exactly when their finds are equal.
        pass

    def union(self, a: int, b: int) -> bool:
        # Merge the groups of a and b. Attach the root of the smaller group under the root of the
        # larger one, update the size and decrease the group count. Return True if a merge
        # happened and False if a and b were already in the same group.
        #
        # Example: on UnionFind(3), union(0, 1) gives True, then union(1, 0) gives False, and
        # the group count is 2.
        pass


def redundant_connection(edges: list[tuple[int, int]]) -> tuple[int, int]:
    # Problem 2: Redundant Connection.
    # Given a list of undirected edges over nodes numbered 1 to n, where the edges form a tree
    # plus exactly one extra edge, return the edge that closes the cycle. Process the edges in
    # order and return the first one whose endpoints are already connected, which is the first one
    # whose union returns False.
    #
    # Example: redundant_connection([(1, 2), (1, 3), (2, 3)]) gives (2, 3).
    # Example: redundant_connection([(1, 2), (2, 3), (3, 4), (1, 4), (1, 5)]) gives (1, 4).
    pass


def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> int | None:
    # Problem 3: Minimum Spanning Tree Weight.
    # Given n nodes numbered 0 to n - 1 and undirected weighted edges (u, v, weight), return the
    # total weight of a minimum spanning tree. If the graph is not connected, return None. When
    # n is 1, return 0.
    #
    # Example: kruskal_mst(4, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 3, 10), (0, 2, 5)]) gives 6
    # using the edges of weight 1, 2 and 3.
    # Example: kruskal_mst(4, [(0, 1, 1), (2, 3, 1)]) gives None.
    #
    # Sort the edges by weight, accept an edge when union returns True, and stop after n - 1
    # accepted edges.
    pass


if __name__ == '__main__':
    # Check that redundant_connection([(1, 2), (1, 3), (2, 3)]) == (2, 3).
    # For kruskal_mst on random graphs with 6 nodes, compare against a brute force that tries every
    # subset of n - 1 edges, keeps those that connect all nodes, and takes the cheapest.
    pass
