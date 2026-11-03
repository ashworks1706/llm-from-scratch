# a run produces one result per (case, suite). the report groups results by suite, in a fixed registry order,
# and keeps passed/total per suite. a suite with no results is left out of the report entirely.
# a rate with zero total is 0.0, not a division error. the same report shape is what gets promoted to the baseline.
#
# from sparky: apps/evals/cli.py (_report, _print, baseline_cmd), SuiteReport.rate in apps/evals/core/types.py,
# apps/evals/baseline.json (suite -> {passed, total}). scorers returning None never become results.
#
# explain: why store passed and total instead of only a rate in the baseline?
# explain: a suite with 1 case flips between 0% and 100%. what does that do to a strict gate?

from dataclasses import dataclass


@dataclass
class Result:
    case_id: str
    suite: str
    passed: bool


@dataclass
class SuiteReport:
    suite: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        # passed / total, 0.0 when total is 0.
        pass


def build_report(results: list[Result], registry_order: list[str]) -> list[SuiteReport]:
    # one SuiteReport per suite that has results, ordered by registry_order.
    # extra, not in sparky: a result naming a suite not in registry_order raises ValueError instead of being dropped.
    # complexity target: O(len(results) + len(registry_order)).
    pass


def promote(report: list[SuiteReport]) -> dict[str, dict[str, int]]:
    # {suite: {"passed": n, "total": m}} for every suite in the report.
    pass


if __name__ == '__main__':
    # asserts to write: empty results -> empty report; ordering follows the registry, not arrival;
    # unknown suite raises; rate of an empty suite is 0.0.
    pass
