# pass@k: probability that at least one of k samples is correct. with n samples per problem and c
# correct, the unbiased estimator is 1 - C(n-c, k) / C(n, k). compute it stably, not with huge factorials.
# averaging per-problem estimates over problems gives the benchmark score.
# contamination: if benchmark text appears in training data the score measures memory. a cheap check
# is n-gram overlap: the share of a test item's n-grams (n around 8-13 tokens) found in the corpus.
# deps: stdlib only. see mlops/04_benchmarking_and_evals.py and mlops/05_llm_eval_and_judges.py.
# explain: why does the naive estimator c/n for pass@1 differ from sampling k=1 once, and which has lower variance?
# explain: why can a model with a high benchmark score still fail on a rephrased question?

def pass_at_k(n: int, c: int, k: int) -> float:
    # 1 - prod_{i=n-c+1}^{n} (1 - k/i). ValueError if k > n or c > n. c == 0 -> 0.0. n - c < k -> 1.0.
    pass


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    # set of contiguous n-grams. fewer than n tokens gives an empty set.
    pass


def overlap_ratio(test_tokens: list[str], corpus_ngrams: set, n: int = 8) -> float:
    # fraction of the test item's n-grams present in corpus_ngrams. empty item gives 0.0.
    pass


def flag_contaminated(items: list[list[str]], corpus_ngrams: set, n: int = 8, threshold: float = 0.5) -> list[int]:
    # indices of items whose overlap_ratio >= threshold.
    pass


if __name__ == '__main__':
    # assert pass_at_k(10, 0, 5) == 0.0, pass_at_k(10, 10, 5) == 1.0, pass_at_k(n, c, 1) == c / n.
    # build a tiny corpus, plant one verbatim test item, assert only it is flagged.
    pass
