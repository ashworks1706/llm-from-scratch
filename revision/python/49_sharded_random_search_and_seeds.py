# stdlib + numpy. stage: method + reproducibility (hyperparameter search on a cluster job array).
# draw ALL trial configs up front from one rng seeded per model; shard k runs trials with trial % num_shards == k,
# so trial i has the same config however you split the work. each finished trial appends one json line to a per-shard file.
# merge: read all shard files, keep the LATEST record per trial (requeued shards redo trials), skip failed ones, pick min val mse.
# a failed trial is recorded and skipped; it must not kill the shard. seed each trial separately (base + trial) for torch/numpy/random.
# sampling rules: choice = pick from list, uniform = flat in [lo, hi], log_uniform = flat in log space (lr gets equal mass per decade).
# from risky_navigation: experiments/random_search.py (sample_value, trial_configs seeded with crc32(model name),
# run_search, merge), scripts/sol_search.sh (SLURM array task id -> shard, array count -> num_shards)
# and scripts/sol_final.sh (merge job submitted with --dependency=afterany);
# experiments/exp_utils.py set_global_seeds. search runs on the tuning group, never on a CV fold.
# explain: why crc32 of the name and not python hash() for the seed? why afterany and not afterok for the merge job?
# explain: random vs grid when only 2 of 6 knobs matter?
import numpy as np


def sample_value(spec: tuple, rng: np.random.Generator):
    # spec is ('choice', [..]) | ('uniform', lo, hi) | ('log_uniform', lo, hi). unknown kind -> ValueError.
    pass


def trial_configs(model: str, space: dict, trials: int, base_seed: int) -> list[dict]:
    # rng seeded from (base_seed, a stable hash of model). draw trials configs in order, same result every run.
    pass


def my_shard(trials: int, shard: int, num_shards: int) -> list[int]:
    # round-robin split; assert 0 <= shard < num_shards. every trial belongs to exactly one shard.
    pass


def merge_records(records: list[dict]) -> dict | None:
    # records have 'trial', 'status', 'val_mse' (maybe from several shard files, duplicates possible, later wins).
    # return the ok record with the lowest finite val_mse, ties broken by lower trial; None if nothing succeeded.
    pass


if __name__ == '__main__':
    # asserts: trial_configs twice is identical; shards for num_shards=1..7 partition range(trials) with no overlap;
    # log_uniform(1e-5, 1e-2) puts about a third of draws in each decade; merge keeps the latest duplicate.
    pass
