# numpy only.
# operator sees the state from d steps ago plus the d commands since. replaying the buffered
# actions from the oldest state under the true dynamics gives the current state exactly.
import numpy as np


# DelayBuffer
# holds at most delay (positive int) (state, action) pairs; pushing when full drops the oldest.
# both queues bounded and self-evicting (no list slicing to trim). replay rebuilds the newest state.
# ex: delay 2, step_fn = state + action, push (state 0, action 1), (state 1, action 2) -> replay 3
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


if __name__ == '__main__':
    # check 1: run a known system for delay steps, pushing each state/action; replay == true state.
    # check 2: push one more step; buffer still holds exactly delay pairs and replay == true state.
    pass
