# pyproject.toml declares the project: name, requires-python, dependencies, [project.scripts] entry points, a build backend.
# an entry point "pkg.mod:func" is just a string resolved with importlib.import_module and getattr at launch time.
# src layout puts code in src/pkg so tests import the INSTALLED package, not whatever sits in the current directory.
# import runs a module once and caches it in sys.modules; a cycle hands the second importer a half-initialized module.
# cycle fixes: import the module instead of names from it, move the shared piece to a third module, or import inside the function.
# from platform: pyproject.toml sets [tool.uv] package = false (an application: uv installs dependencies only) and requires-python >=3.12,<3.13.
# from platform: pyproject.toml [tool.importlinter] contracts forbid core from importing modules, so cycles upward cannot form.
# from platform: modules/manifest.py _load(names) calls importlib.import_module on each name so models and jobs register themselves.
# explain: what does uv sync do that pip install -r does not, and why does a lockfile matter for a deployed app?
# explain: why does from a import x fail in a cycle when import a would work?

import importlib
from collections.abc import Callable
from types import ModuleType


def parse_entry_point(spec: str) -> tuple[str, str]:
    # "pkg.mod:func" -> ("pkg.mod", "func"); the attribute may be dotted ("pkg.mod:Cls.run").
    # raise ValueError when the colon is missing, repeated, or either side is empty or has whitespace.
    pass


def resolve_entry_point(spec: str) -> Callable:
    # import the module and walk the dotted attribute path with getattr; the result must be callable, else TypeError.
    # a missing module raises ModuleNotFoundError; a missing attribute raises AttributeError naming the spec.
    pass


def load_all(names: list[str]) -> dict[str, ModuleType]:
    # import every name (like platform's manifest loader), but gather ALL failures and raise one ImportError listing them.
    # one broken module must not hide the others' errors.
    pass


def find_import_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    # graph maps module -> modules it imports. return one cycle as a path that starts and ends on the same node, or None.
    # DFS with white/grey/black states; a grey neighbour closes a cycle. O(V + E). ignore nodes missing from the graph keys.
    pass
