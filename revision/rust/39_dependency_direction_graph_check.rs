// layering rules are a graph problem. between apps: engine, discord and cli may not depend on each other
// (scripts/check-deps.sh fails on any in-repo path dependency). inside the engine: core imports nothing
// else in the crate, harness/model/tools/stores import only core and never each other, wiring/routes compose them.
// a checker is: list edges, reject the ones the rule forbids, and detect cycles for good measure.
//
// from sparky: scripts/check-deps.sh (cargo metadata, jq over path dependencies) and the module rules
// in AGENTS.md. the engine rule is "checked in review" there, so writing it as code is the exercise.
//
// explain: why enforce between-app isolation by script but the in-crate layering by review? what would a
// compiler-level fix look like (separate crates)?
// explain: a cycle between two allowed modules passes the pairwise rule; which check catches it?

use std::collections::{HashMap, HashSet};

fn app_violations(path_deps: &[(&str, &str)]) -> Vec<String> {
    // every (package, dependency) pair is a violation: apps may not depend on each other in-repo.
    // output "a -> b" strings in input order.
    todo!()
}

fn module_violations(imports: &[(&str, &str)]) -> Vec<(String, String)> {
    // modules: core, harness, model, tools, stores, wiring, routes.
    // allowed edges: anything -> core (not core -> anything else); wiring and routes -> any module.
    // harness, model, tools, stores -> only core. same-module imports are fine.
    // return the forbidden (from, to) edges.
    todo!()
}

fn has_cycle(edges: &[(&str, &str)]) -> bool {
    // directed cycle detection in O(V + E): dfs with visiting / done sets. a self edge counts as a cycle.
    todo!()
}

fn main() {
    // tests: core -> harness is a violation; harness -> model is a violation; wiring -> harness is fine;
    // a -> b -> c -> a is a cycle; a diamond is not.
}
