# stdlib + numpy. stage: result and limitations (bugs that silently change what an experiment means).
# assignment inside a function creates a LOCAL name unless declared global; reads fall through to module scope.
# in-place numpy ops (x -= m, x[...] = ...) mutate every alias; fancy indexing copies, basic slicing views.
# temporarily mutating self state without try/finally leaves it changed if the body raises.
# from risky_navigation: src/learning/SAC/sac_agent_CleanRL.py has module-level _cpt used by risk_measure(), and main()
# assigns _cpt = CumulativeProspectTheory(...) with no global, so the real CPT object is local and the critic always
# used the all-ones one (a risk-neutral mean). see also CumulativeProspectTheory.sample_expectation_batch in
# src/learning/risk_measures.py (swaps self.b, restores only on the success path) and experiments/data_pipeline.py
# make_fold (val and test are .copy()ed before _standardize edits them in place, so the stored partitions stay raw).
# explain: how would you detect the _cpt bug without reading the code? (hint: a test on behavior, not on names)
# explain: what two ways fix it, and why is a config object passed to risk_measure better than a global?
import numpy as np

_risk = {'name': 'mean'}  # module-level default used by risk_value below


def risk_value(samples: np.ndarray, risk=None) -> np.ndarray:
    # use the module default when risk is None; 'mean' -> row mean, 'worst' -> row min. [B, N] -> [B].
    pass


def configure_wrong(name: str) -> None:
    # reproduce the bug: assign a new dict to the module-level name WITHOUT global; show risk_value ignores it.
    pass


def configure_right(name: str) -> None:
    # make the change visible to risk_value on purpose (say which of the two fixes you chose in a comment).
    pass


def standardize_without_alias_bug(folds: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    # return standardized arrays and leave the inputs untouched; know when in-place is safe (owned copy) and when not.
    pass


def with_temp_attr(obj, name: str, value, fn):
    # set obj.name = value, call fn(), always restore the old value, including when fn raises.
    pass


if __name__ == '__main__':
    # asserts: configure_wrong leaves risk_value unchanged; configure_right changes it; inputs of the standardizer are
    # equal to a saved copy afterwards; with_temp_attr restores after an exception.
    pass
