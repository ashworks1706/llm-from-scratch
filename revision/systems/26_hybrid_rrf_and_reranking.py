# lexical search (BM25) finds exact terms, ids and rare words; dense search finds paraphrases. their
# scores live on different scales, so do not add them raw. reciprocal rank fusion uses ranks only:
# score(d) = sum over lists of 1 / (k + rank_in_list), k around 60, rank starting at 1. a doc missing
# from a list contributes nothing from it.
# a typical pipeline: cheap recall stage (top ~50 from each leg) -> fuse -> expensive reranker
# (cross-encoder or llm) on the top few -> cut to top_k. the reranker only helps if recall already
# put the answer in the candidates.
# from platform: modules/knowledge/search.py rrf(lists, k=60.0) fuses the vector leg and the text leg,
# using 1 / (k + rank + 1) over a 0-based enumerate, with CANDIDATES = 50 per leg and tuning["rrf_k"].
# from sparky: apps/engine/src/stores/platform/retrieval.rs calls the platform search; fusion happens there.
# explain: why does rrf need no score normalization, and what does it throw away?
# explain: what recall@50 vs precision@5 tradeoff does adding a reranker change?

def rrf(lists: list[list[str]], k: float = 60.0) -> list[tuple[str, float]]:
    # fused (id, score) sorted by score desc; ties broken by id so output is deterministic.
    # duplicates inside ONE list count once (first position). empty lists are fine.
    pass


def rerank(candidates: list[str], scorer, top_n: int) -> list[str]:
    # scorer(doc_id) -> float (a stand-in for a cross-encoder). only the first top_n candidates are
    # scored (cost cap); the rest keep their original relative order after them.
    pass


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    pass


def mrr(ranked_lists: list[list[str]], relevant: list[set[str]]) -> float:
    # mean reciprocal rank of the first relevant hit per query (0 when none).
    pass


if __name__ == '__main__':
    # build a case where leg A and leg B each miss a different relevant doc; assert the fused list
    # recalls both at k=3 and a raw score sum would not. assert rrf of one list preserves its order.
    pass
