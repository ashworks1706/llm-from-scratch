# tool scorers read the engine trace, not the answer text: which tool ran, and with what arguments.
# a scorer returns None when the case has nothing for it to judge, so the case is not counted at all.
# None and "passed=False" are different: None keeps the pass rate honest, False is a real failure.
# arguments in a trace may arrive as a dict or as a json string; both must be handled.
#
# from sparky: apps/evals/suites/tool_selection.py and tool_args.py, TurnResult.tool_calls in
# apps/evals/core/types.py (events with kind "tool_call"), core/tests/test_suites.py for the expected behavior.
#
# explain: why match the expected value INSIDE json.dumps(args[key]) lowercased, not by equality?
# explain: tool_selection with no expected tool passes when nothing was called. what regression does that catch?

import json


def tool_calls(turns: list[dict]) -> list[dict]:
    # every event with kind == "tool_call" across all turns, in order.
    # a turn is {"events": [ {...}, ... ]}.
    pass


def score_selection(expected_tool: str | None, turns: list[dict]) -> tuple[bool, str]:
    # no expected tool: pass only if no tool was called. else pass if expected_tool is among the called.
    # the detail string names what was expected and what was called, in call order.
    pass


def score_args(expected_tool: str | None, contains: dict[str, str], turns: list[dict]) -> tuple[bool, str] | None:
    # None when contains is empty. fail if the expected tool was never called.
    # pass if ANY call of that tool has every key present with value.lower() in json.dumps(args[key]).lower().
    # args that are a json string get json.loads; not a dict after that, or undecodable -> skip and count it.
    # when failing, mention how many calls had unreadable arguments.
    pass


if __name__ == '__main__':
    # asserts to write: key present but wrong value; wrong key name fails; string arguments;
    # an unreadable argument string does not crash; empty contains returns None.
    pass
