# value iteration applies the bellman optimality backup until values stop changing:
# V(s) = max_a sum_s' P(s'|s,a) [r + gamma V(s')]. it needs the model (P and r) and converges for gamma < 1.
# terminal states hold value 0 and are never updated; walls keep you in place.
# the greedy policy read off V is the output. check the stopping rule: max change < tol.
# deps: stdlib or numpy. grid is a list of strings: '.' free, '#' wall, 'G' goal (+1, terminal), 'X' trap (-1, terminal).
# explain: why does gamma = 1 with a step cost still terminate, and what breaks if there is no terminal state?
# explain: value iteration vs policy iteration: which does more work per sweep and which needs fewer sweeps?

ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def step(grid: list[str], pos: tuple[int, int], action: str) -> tuple[int, int]:
    # deterministic move. out of bounds or into a wall returns pos unchanged.
    pass


def value_iteration(grid: list[str], gamma: float = 0.95, step_cost: float = -0.04, tol: float = 1e-6) -> dict:
    # return {(r, c): V} for every non-wall cell. reward is on ENTERING G or X; each other move costs step_cost.
    # synchronous sweeps (compute from the old V). bounded: raise if it exceeds 10_000 sweeps.
    pass


def greedy_policy(grid: list[str], V: dict, gamma: float = 0.95, step_cost: float = -0.04) -> dict:
    # {(r, c): action letter} for non-terminal cells. break ties in the order U, D, L, R.
    pass


if __name__ == '__main__':
    # 3x4 grid: ["...G", ".#.X", "...."]. assert V at G-adjacent cells is the largest,
    # the policy never walks into X when a safe route exists, and rerunning from the final V changes nothing.
    pass
