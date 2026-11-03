# next-token loss: logits at position t predict the token at t+1. so drop the last logit and the
# first label (shift), then cross entropy over the remaining positions.
# ignore_index (usually -100) removes positions from BOTH the sum and the denominator: padding and
# prompt tokens must not count. an all-ignored batch gives nan with mean reduction.
# the right normalizer is the count of supervised tokens, not batch*seq. across microbatches it
# must be the global token count or short sequences get overweighted.
# deps: torch. see sft/train.py and sft/dataset.py (instruction tokens are masked there).
# explain: what silently happens if you forget the shift and train anyway?
# explain: why should perplexity be exp of mean loss over TOKENS and not over sequences?

import torch

IGNORE = -100


def shifted_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits (b,t,v), labels (b,t) with IGNORE where unsupervised. shift, flatten, cross entropy.
    # return the mean over supervised tokens; return 0.0 (with grad path intact) if there are none.
    pass


def manual_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # same result as shifted_lm_loss written with log_softmax + gather + mask, no F.cross_entropy.
    pass


def perplexity(total_nll: float, n_tokens: int) -> float:
    # exp of the token-mean nll. n_tokens == 0 raises ValueError.
    pass


if __name__ == '__main__':
    # assert manual_cross_entropy == shifted_lm_loss on random data with some IGNORE labels.
    # assert changing a logit at an ignored position does not change the loss or the grads elsewhere.
    pass
