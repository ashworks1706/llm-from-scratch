# rlhf-style rewards subtract a kl penalty to a frozen reference policy: r_total = r - beta * KL(pi || ref).
# per token, a sampled estimate is logp_pi(token) - logp_ref(token) on the tokens the policy produced.
# the penalty keeps the policy near what the reference model can sensibly say.
# reward hacking: the policy finds inputs the reward model scores high that are not actually good
# (length padding, flattery, repeated phrases, exploiting a bug in the scorer).
# deps: torch. see rl/dpo_trainer.py, which handles the reference model with beta inside the loss.
# explain: give two reward hacks for a summarization reward model and one measurable signal for each.
# explain: what happens to quality and to kl as beta goes to 0, and as beta goes to infinity?

import torch


def sampled_kl(logp_pi: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # (b,t) log probs of the sampled tokens. return per-sequence summed (logp_pi - logp_ref) over masked tokens, shape (b,).
    pass


def shaped_reward(task_reward: torch.Tensor, kl: torch.Tensor, beta: float) -> torch.Tensor:
    # task_reward and kl are (b,). return task_reward - beta * kl.
    pass


def kl_k3(logp_pi: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    # low-variance non-negative estimator: with r = exp(logp_ref - logp_pi), per-token (r - 1) - log(r).
    pass


if __name__ == '__main__':
    # assert sampled_kl is 0 when policy == reference and masked-out tokens do not contribute.
    # assert kl_k3 >= 0 elementwise; compare its mean to the exact kl on two small categorical distributions.
    # toy hack: give a reward that grows with length, show the optimum length rises as beta shrinks.
    pass
