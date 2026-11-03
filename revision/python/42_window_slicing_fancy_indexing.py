# numpy only. stage: method (table rows -> model inputs). one start row s defines one sample.
# history rows are s, s+dL, ..., s+L-dL; the current row is s+L; the label is the action at row s+L+offset.
# build a whole batch at once: starts [B] + offsets [S] broadcast to an index grid [B, S], then one gather.
# flat layout (mlp, vae) concatenates history and appends the current OBSERVATION only (no action, it is the answer).
# token layout (transformer, vq-bet) appends a final token [current obs | zeros for the action slot].
# from risky_navigation: src/learning/SAC/sac_agent_demonstrate.py _build_sequences and _reshape_predictors.
# row layout there: obs columns, then action columns, then the terminal flag last (sliced off with :-1).
# explain: for L=30, dL=10 what are flat dim and token shape with obs_dim=30, act_dim=2? and for L=6, dL=6?
# explain: why is a python loop over B starts a bad idea, and what does fancy indexing copy vs view?
import numpy as np


def history_index_grid(starts: np.ndarray, L: int, dL: int) -> np.ndarray:
    # return int64 [B, S] with S = L // dL, grid[b, j] = starts[b] + j * dL. no python loop.
    pass


def build_windows(table: np.ndarray, starts: np.ndarray, L: int, dL: int, offset: int, obs_dim: int, act_dim: int, flat: bool):
    # table is [N, obs_dim + act_dim + 1]. return (inputs, labels).
    # flat -> inputs [B, S*(obs+act) + obs]; tokens -> inputs [B, S+1, obs+act] with zero action in the last token.
    # labels [B, act_dim] from row starts + L + offset. keep float32 end to end.
    pass


if __name__ == '__main__':
    # build a tiny table where every cell encodes its own row index; assert history rows, current row and label row
    # are exactly the ones you expect, for two (L, dL) pairs, flat and tokens, offset 0 and offset 6.
    pass
