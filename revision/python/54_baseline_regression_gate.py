# the gate compares this run to a committed baseline and fails if any suite's pass rate fell.
# tolerance is an allowed drop in rate per suite (default 0). rules: a suite in the baseline but missing
# from the run is a regression; a suite in the run but not in the baseline (or with total 0 there) is
# reported and not compared; improvements never fail. the gate exits non-zero on any regression.
#
# from sparky: apps/evals/cli.py (compare_cmd), apps/evals/baseline.json, and the just eval
# recipes (run, baseline, compare). baseline is {suite: {"passed": int, "total": int}}.
#
# explain: with a baseline of 1/3 on a suite and a run of 1/2, is that a regression? what is the sampling problem?
# explain: why must a MISSING suite fail the gate, instead of being skipped like a new suite?

def compare(baseline: dict[str, dict[str, int]], now: dict[str, tuple[int, int]], tolerance: float = 0.0) -> tuple[list[str], list[str]]:
    # now maps suite -> (passed, total). return (regressions, notes).
    # iterate the sorted union of suite names so output order is stable.
    # regression text: "<suite>: <before:.0%> -> <after:.0%>" when after + tolerance < before.
    # missing from now -> "<suite>: in the baseline but not in this run".
    # not in baseline or baseline total == 0 -> a note, no comparison.
    # compare as floats but be careful with equality: after + tolerance < before, strict.
    pass


def exit_code(regressions: list[str]) -> int:
    # 1 when there is any regression, else 0.
    pass


if __name__ == '__main__':
    # asserts to write: equal rates pass; a drop of exactly the tolerance passes; a drop larger than
    # tolerance fails; a missing suite fails; a new suite is only a note; an improvement passes.
    pass
