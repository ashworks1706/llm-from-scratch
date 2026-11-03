# an mcp or /api/tools call goes through one pipeline. org active, rate limit, tool visible, args valid,
# confirm step, run, then audit in a finally so refused calls are logged too.
# unknown tool and not-allowed tool give the same 404, so a token cannot probe which tools exist.
# a confirm tool without confirm=true runs nothing and returns a preview. a batch is many calls, each fully checked.
# from platform: modules/mcp/runtime.py call(), _usable(), _preview(), _batch() (MAX_BATCH 25, no nested batch).
# from platform: core/tools.py ToolSpec(scope, module, confirm, input_schema) and the @tool decorator.
# explain: why does the rate limit consume before the tool lookup, and what does that cost an attacker?
# explain: why does the audit row for a 403 omit the org prefix?

from dataclasses import dataclass


class ToolError(Exception):
    def __init__(self, message: str, status: int):
        pass


@dataclass(frozen=True)
class Spec:
    name: str
    scope: str
    module: str | None
    confirm: bool
    func: object


def visible(specs: dict[str, Spec], scopes: set[str], modules_on: set[str]) -> list[str]:
    # names the token may call: its scope is held and (module is None or module in modules_on). sorted.
    pass


def call(specs: dict[str, Spec], scopes: set[str], modules_on: set[str], name: str, args: dict, audit: list) -> object:
    # unknown or not visible -> ToolError 404 (same message). pop "confirm" from args.
    # spec.confirm without confirm is True -> return {"confirm_required": True, "tool": name} and run nothing.
    # else return spec.func(**args). append (name, status) to audit exactly once per call, even on error.
    pass


def batch(specs: dict[str, Spec], scopes: set[str], modules_on: set[str], calls: list[dict], stop_on_error: bool, audit: list) -> list[dict]:
    # 1..25 calls else ToolError 400. a call named "batch" fails with 400 for that item only.
    # each result {"tool", "ok", "status"}. stop at the first failure only when stop_on_error.
    pass


if __name__ == '__main__':
    # assert: a delete tool without confirm does not run, unknown vs forbidden are indistinguishable,
    # a 404 call is still audited, a batch with stop_on_error halts at the first failure.
    pass
