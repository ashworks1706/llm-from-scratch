# numpy only.
# an operator sees the robot state from d steps ago plus the d commands sent since.
# two fixed-length fifo queues hold past states and the action applied from each state.
# replaying the buffered actions from the oldest state under the true dynamics gives the current state exactly.
import numpy as np


class DelayBuffer:
    def __init__(self, delay: int):
        # two bounded fifos of length delay for states and actions, with no list slicing to trim.
        pass

    def push(self, state: np.ndarray, action: np.ndarray) -> None:
        # append the pair and let the oldest pair fall out once the buffer is full.
        pass

    def replay(self, step_fn) -> np.ndarray:
        # fold step_fn over the buffered actions starting from the oldest state.
        # return the inferred current state, which is only valid when the buffer is full.
        pass


if __name__ == '__main__':
    # after d pushes, assert replay equals the true state tracked outside the buffer.
    # push one more step and assert the buffer still has length d and replay still matches.
    pass
