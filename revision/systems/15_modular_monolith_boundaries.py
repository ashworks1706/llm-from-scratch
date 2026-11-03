# monolith: one deployable, in-process calls, one transaction. microservices: separate deployables,
# network calls, independent scaling and release, and every call can fail or be slow. a modular
# monolith keeps one deployable but enforces module boundaries (a public surface, no reaching into
# internals, a one-way dependency rule), so extraction later is cheap.
# from platform: pyproject.toml configures import-linter ("core imports no module", "service modules do
# not import Flask"), modules/registry.py mounts each module's blueprint, and AGENTS.md says core/ imports
# nothing from modules/. build a toy checker for those two rule kinds.
# explain: what do you give up and gain moving one module to its own service (transactions, latency, deploys)?

import ast


def imports_of(source: str) -> set[str]:
    # top-level dotted module names imported by python source (ast.walk, Import and ImportFrom).
    # relative imports (level > 0) resolve against package only if you add that param; skip them here.
    pass


def forbidden_edges(files: dict[str, str], rules: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    # files maps a module path like "core.cache" to its source. rules are (source_prefix, forbidden_prefix).
    # return (file, rule_source, imported_module) for every violation. "core" must match "core.cache"
    # but NOT "core2". sort the output for determinism.
    pass


def import_cycles(files: dict[str, str]) -> list[list[str]]:
    # find cycles in the module dependency graph restricted to modules in files. each cycle once,
    # rotated to start at its smallest name. dfs with colors; do not recurse past 1000 modules.
    pass


if __name__ == '__main__':
    # assert core importing modules.points is flagged and modules.points importing core is not.
    # assert a -> b -> a is one cycle.
    pass
