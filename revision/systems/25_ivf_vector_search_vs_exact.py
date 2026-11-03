# exact search scores the query against EVERY vector: O(N*d) per query, perfect recall.
# IVF clusters vectors with k-means into nlist cells; a query scans only the nprobe nearest cells, about
# N*nprobe/nlist work, but recall drops when the true neighbor sits in an unprobed cell.
# HNSW is another family: a layered proximity graph searched greedily top to bottom; high recall and fast,
# but more memory and slower inserts. product quantization compresses vectors into codes. requires numpy.
# from piramid: docs/ROADMAP.md lists HNSW and IVF under Out of scope; apps/engine/database/src/search/engine.rs
# scores every stored vector and keeps the best k. this is interview knowledge to contrast with that exact
# scan. from platform: modules/knowledge/search.py _dense_python also scores every stored vector (cosine).
# explain: at what N does an exact scan stop being acceptable, and when does a GPU extend that?

import numpy as np


def exact_topk(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    # indices of the k nearest by L2 distance, nearest first. use argpartition then sort the k.
    pass


def kmeans(data: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    # return centroids (k, d). init by choosing k distinct data points. handle an empty cluster by
    # re-seeding it with a random point. deterministic for a fixed seed.
    pass


class IVF:
    def __init__(self, nlist: int, seed: int = 0):
        pass

    def train_and_add(self, data: np.ndarray) -> None:
        # kmeans for centroids, then assign each vector to its nearest centroid: lists[c] = array of ids.
        pass

    def search(self, query: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        # nearest nprobe centroids, candidate ids from those lists, exact rerank on candidates.
        pass


def recall_at_k(approx: np.ndarray, truth: np.ndarray) -> float:
    # |approx intersect truth| / |truth|, averaged over queries by the caller.
    pass


if __name__ == '__main__':
    # sweep nprobe in 1,2,4,8,16 and print recall@10 and the fraction of vectors scanned.
    # assert recall is non-decreasing in nprobe and equals 1.0 when nprobe == nlist.
    pass
