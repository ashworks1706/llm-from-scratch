# lexical search (BM25) finds exact terms and ids; dense search finds paraphrases. their scores are on
# different scales, so fuse by rank: reciprocal rank fusion score(d) = sum over lists of 1 / (k + rank),
# k around 60. a doc missing from a list adds nothing from it.
# pipeline: cheap recall (top ~50 per leg) -> fuse -> expensive reranker (cross-encoder or llm) on the top few.
# a reranker only helps if recall already put the answer in the candidates.
# from platform: modules/knowledge/search.py rrf(lists, k=60.0) fuses the vector and text legs with
# 1 / (k + rank + 1) over a 0-based enumerate; CANDIDATES = 50 per leg; k comes from tuning["rrf_k"].
# from sparky: apps/engine/src/stores/platform/retrieval.rs calls platform search, which fuses.
# explain: why does rrf need no score normalization, and what information does it throw away?

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
