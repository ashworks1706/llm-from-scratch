# rl in one loop: observe state, pick action from a policy, get reward and next state, update.
# on-policy (ppo, reinforce) learns from data its current policy produced; off-policy (sac, q-learning,
# dpo-style offline) can reuse old data. llm post-training uses the same objects with tokens as actions.
# pointers: rl/dpo_trainer.py is written (reread it). rl/ppo_trainer.py and rl/grpo_trainer.py are short
# scaffolds: solve them AFTER 04, 05 and 06 in this folder, then come back and fill the trainers.
# from risky_navigation: src/learning/SAC/sac_agent_CleanRL.py has Actor.get_action (tanh-squashed
# gaussian with a log_prob correction), twin SoftQNetwork critics, gamma/tau/alpha in Args, and entropy
# autotune (log_alpha, target_entropy). read it for the off-policy shape, not to rebuild it.
# explain: why do sac critics take min over two q networks, and what does overestimation do to the actor?
# explain: where does a replay buffer help and where does it hurt an on-policy method?

import math


def tanh_squash_logprob(logp_gaussian: float, u: float) -> float:
    # sac squashes an unbounded sample u with tanh. the density changes by |d tanh(u)/du| = 1 - tanh(u)^2.
    # return the corrected log prob of a = tanh(u): logp_gaussian - log(1 - a^2 + eps), eps = 1e-6.
    pass


def soft_target(reward: float, done: bool, gamma: float, min_q_next: float, alpha: float, logp_next: float) -> float:
    # sac critic target: r + (1 - done) * gamma * (min_q_next - alpha * logp_next).
    pass


def polyak(target: list[float], online: list[float], tau: float) -> list[float]:
    # soft target update: (1 - tau) * target + tau * online, elementwise, returns a new list.
    pass
