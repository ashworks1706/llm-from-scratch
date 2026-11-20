# numpy only.
# operator sees the state from d steps ago plus the d commands since. replaying the buffered
# actions from the oldest state under the true dynamics gives the current state exactly.
import numpy as np


# DelayBuffer
# holds at most delay (positive int) (state, action) pairs; pushing when full drops the oldest.
# both queues bounded and self-evicting (no list slicing to trim). replay rebuilds the newest state.
class DelayBuffer:
    # two bounded queues (states, actions), both maxlen delay.
    def __init__(self, delay: int):
        pass

    # add a state and the action applied from it (numpy arrays). full -> oldest pair dropped.
    def push(self, state: np.ndarray, action: np.ndarray) -> None:
        pass

    # step_fn(state, action) -> next state. start at the oldest state, apply each buffered action
    # oldest first, return the result array. only meaningful once full (callers push delay pairs).
    def replay(self, step_fn) -> np.ndarray:
        pass


# ---- try it ----
if __name__ == '__main__':
    # buf = DelayBuffer(2)
    # buf.push(np.array([0.]), np.array([1.]))
    # buf.push(np.array([1.]), np.array([2.]))
    # assert buf.replay(lambda s, a: s + a) == [3.]
    pass
