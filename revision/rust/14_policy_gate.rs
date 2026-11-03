// every tool declares a risk class; a policy turns (risk, who is asking, where) into a decision.
// RiskClass is ordered, so rules read as thresholds: ">= ExternalWrite needs a write role".
// rule ORDER matters: forbidden is checked first, confirm is checked after the deny rules.
// a denial is not an error: its reason goes back to the model as a tool result.
//
// from sparky: apps/engine/src/runtime/harness/safety/policy.rs (RiskPolicy::authorize),
// core/types/tools/mod.rs (RiskClass), core/types/safety/policy.rs (Decision).
//
// explain: why check the role before the confirm threshold, not after?
// explain: why is "authenticated read" denied in a public channel even if the user holds a role?

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RiskClass { ReadPublic, ReadAuthenticated, PrepareWrite, ExternalWrite, Destructive, Forbidden }

#[derive(Debug, Clone, PartialEq, Eq)]
enum Decision { Allow, Deny(String), Confirm }

struct Ctx { roles: Vec<String>, private: bool }

struct RiskPolicy { write_roles: Vec<String>, allow_authenticated_reads: bool, confirm_from: RiskClass }

impl RiskPolicy {
    fn may_write(&self, ctx: &Ctx) -> bool {
        // true when the ctx holds any role in write_roles. empty write_roles means nobody.
        todo!()
    }

    fn authorize(&self, ctx: &Ctx, risk: RiskClass) -> Decision {
        // order: Forbidden -> Deny; ReadAuthenticated outside a private place (unless the
        // config allows it) -> Deny; risk >= ExternalWrite without a write role -> Deny;
        // risk >= confirm_from -> Confirm; otherwise Allow.
        // note Forbidden is the largest variant, so it would also trip the >= rules: order saves you.
        todo!()
    }
}

fn main() {
    // table-test every RiskClass against {private, public} x {with role, without role}.
}
