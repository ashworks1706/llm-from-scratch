# q-learning is model-free: it learns Q(s,a) from sampled transitions with the td update
# Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).
# the max makes it off-policy: it learns the greedy policy while behaving epsilon-greedily.
# at terminal transitions the bootstrap term is 0. forgetting this is the classic bug.
# exploration: epsilon decays, but never let it hit exactly 0 during training.
# deps: stdlib or numpy. reuse the grid and step() idea from 02_gridworld_value_iteration.py.
# explain: sarsa vs q-learning on a cliff-edge grid: which learns the riskier path and why?
# explain: why can tabular q-learning not scale to images, and what replaces the table?

import random


def epsilon_greedy(q_row: list[float], eps: float, rng: random.Random) -> int:
    # random action with prob eps, else argmax with random tie-break (so a zero table is not biased to action 0).
    pass


def td_update(Q: dict, s, a: int, r: float, s2, done: bool, alpha: float, gamma: float, n_actions: int) -> float:
    # in-place update of Q[s][a]; Q is a dict state -> list of action values (create rows lazily).
    # return the td error. done means no bootstrap.
    pass


def train(env_step, env_reset, n_actions: int, episodes: int, alpha=0.1, gamma=0.95,
          eps_start=1.0, eps_end=0.05, seed=0) -> dict:
    # env_reset() -> state; env_step(state, action) -> (next_state, reward, done). cap each episode's length.
    # eps decays linearly over episodes. return the Q table.
    pass


if __name__ == '__main__':
    # run on a small corridor or grid, then assert the greedy policy matches value iteration's policy.
    pass
