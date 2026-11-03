# sft: next-token loss on response tokens only; prompt tokens are masked out of loss.
# dpo: for a (prompt, chosen, rejected) triple use sequence logprobs under policy and frozen reference:
# loss = -logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r))). sequence logprob = sum over RESPONSE tokens.
# grpo: sample a group of k responses per prompt, advantage = (reward - group mean) / (group std + eps).
# no value network; the group is the baseline.
# deps: torch. reread rl/dpo_trainer.py (compute_log_probs, train_step) and sft/dataset.py first.
# rl/grpo_trainer.py and rl/ppo_trainer.py are still scaffolds: solve those after these stubs.
# explain: why does dpo need the reference logprobs precomputed or computed under no_grad?
# explain: what happens to grpo advantages when every sample in a group gets the same reward?

import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor, ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    # logits (b,t,v), ids (b,t), response_mask (b,t) 1 on response tokens. return (b,).
    # token logprob at position t comes from logits at t-1. sum only masked positions.
    pass


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    # inputs are (b,) sequence logprobs. return (mean loss, implicit reward margin per example).
    pass


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # rewards (n_prompts, k). normalize within each row (group). identical rewards give zeros, not nan.
    pass


if __name__ == '__main__':
    # assert dpo_loss equals log(2) when policy == reference.
    # assert group_advantages rows have mean ~0 and a constant-reward row is all zeros.
    pass
