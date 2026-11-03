# a golden case is one json object per line: id, suites, question, and an expect block.
# loading is where bad data should die: blank lines are skipped, a case naming an unknown suite
# is a hard error that names the case, the file and the known suites, and zero cases is an error.
# a typo in a suite name must never silently become a case that scores nothing.
# stdlib only: use json and a plain dict for the case. sparky uses pydantic models for this.
#
# from sparky: apps/evals/runner.py (load_cases), apps/evals/core/types.py (EvalCase, Expectation,
# RunnerError), apps/evals/cases/asu_golden.jsonl and tool_routing.jsonl, apps/evals/suites/__init__.py (SUITES).
#
# explain: why fail the whole run on one bad case instead of skipping it with a warning?
# explain: what does "every field in expect is optional" imply for how each scorer must start?

import json


class RunnerError(RuntimeError):
    pass


def parse_case(line: str) -> dict:
    # json.loads the line, require keys id (str), suites (list[str]) and question (str).
    # missing key -> RunnerError naming the key. give expect a default of {} when absent.
    pass


def load_cases(files: dict[str, str], known_suites: set[str]) -> list[dict]:
    # files maps file name -> file text. iterate file names in sorted order, lines in order.
    # skip blank lines. for each case, unknown = sorted(set(suites) - known_suites); if any,
    # raise RunnerError mentioning the case id, the file name and the known suites.
    # no cases at all -> RunnerError. extra, not in sparky: duplicate case ids -> RunnerError too (an id keys the report).
    pass


if __name__ == '__main__':
    # asserts to write: blank lines skipped; unknown suite names the case and file; an empty
    # input raises; two files load in sorted file order; a duplicate id raises.
    pass
