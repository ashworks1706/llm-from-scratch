# numpy only.
# An operator sees the robot state from d steps ago plus the d commands sent since then. Two fixed
# length first in first out queues hold the past states and the action applied from each state.
# Replaying the buffered actions from the oldest state under the true dynamics gives the current
# state exactly.
import numpy as np


# Problem: delay buffer with replay.
#
# DelayBuffer remembers the last delay pairs of a state and the action applied from that state.
# delay is a positive int. The buffer holds at most delay pairs, so adding a new pair once it is
# full drops the oldest pair. Both queues must be bounded and drop old entries on their own, with
# no list slicing to trim them. replay uses the dynamics function to rebuild the newest state.
#
# Example: with delay 2, step_fn(state, action) = state + action, push (state 0, action 1) and then
# (state 1, action 2). replay gives 3, because starting from the oldest state 0 it applies 1 and
# then 2.
class DelayBuffer:
    # Create the two bounded queues, one for states and one for actions, both of length delay.
    def __init__(self, delay: int):
        pass

    # Add the state and the action applied from it, both numpy arrays. When the buffer already
    # holds delay pairs, the oldest pair is removed. Nothing is returned.
    def push(self, state: np.ndarray, action: np.ndarray) -> None:
        pass

    # Take step_fn, a function of (state, action) that returns the next state. Start from the
    # oldest buffered state and apply step_fn with each buffered action in order, oldest first.
    # Return the resulting state as an array. The result is only meaningful when the buffer is
    # full, so callers push delay pairs first.
    def replay(self, step_fn) -> np.ndarray:
        pass


if __name__ == '__main__':
    # Check one: run a known system for delay steps, pushing each state and action, and assert that
    # replay equals the true state tracked outside the buffer.
    # Check two: push one more step and assert the buffer still holds exactly delay pairs and that
    # replay still equals the true state.
    pass
