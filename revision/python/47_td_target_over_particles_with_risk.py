# numpy only. stage: method (what the demonstrator optimizes).
# for one (s, a) the simulator imagines N next outcomes (N dynamics-parameter draws, all fed the same buffered actions).
# each outcome i gives a td target y_i = r_i + gamma * (1 - d_i) * (min(Q1', Q2')_i - alpha * logpi_i) with ITS OWN reward, done, next obs.
# the critic regresses to one scalar per row: risk(y_1..y_N). the mean is risk-neutral; a cpt-style rank weighting inflates the tails.
# target networks track the live ones by polyak averaging: theta' <- theta' + tau * (theta - theta').
# from risky_navigation: src/learning/SAC/sac_agent_CleanRL.py critic update (next_q_value_samples = r + (1-d)*gamma*min_q_next,
# then risk_measure), torch._foreach_lerp_ target update; src/env/mdp.py resample_dynamics draws the N parameter sets;
# src/learning/risk_measures.py CumulativeProspectTheory.sample_expectation_batch sorts samples per row.
# explain: why min over twin critics, and why is the entropy term subtracted inside the sample, not after the risk reduction?
# explain: what does a (1 - d) mask do if the real robot crashed but only some particles did?
import numpy as np


def predict_particles(x_delayed: np.ndarray, actions: np.ndarray, params: np.ndarray, step_fn) -> np.ndarray:
    # x_delayed [4], actions [d, 2] (oldest first), params [N, P] one dynamics draw per particle.
    # replay all d actions for all N particles at once (vectorized over N, loop only over d). return [N, 4].
    # this is the predict half of a particle filter: no sensor update, so the cloud never sharpens.
    pass


def td_target_samples(r: np.ndarray, d: np.ndarray, q1_next: np.ndarray, q2_next: np.ndarray, logpi_next: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    # all inputs [B, N]. return [B, N] per-particle targets using the pessimistic twin and entropy bonus.
    pass


def rank_weighted_value(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # y [B, N], weights [N] summing to 1 applied to values sorted ascending within each row. return [B].
    # weights = 1/N each must reproduce y.mean(axis=1).
    pass


def soft_update(target: list[np.ndarray], live: list[np.ndarray], tau: float) -> None:
    # in place on target arrays; tau=0 keeps target, tau=1 copies live. do not rebind the list entries.
    pass


if __name__ == '__main__':
    # assert uniform weights equal the mean; put more weight on the lowest ranks and assert the value drops (risk-averse).
    # assert soft_update mutated the original arrays (check their ids/contents), not copies.
    pass
