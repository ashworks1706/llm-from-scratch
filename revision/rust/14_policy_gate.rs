// every tool declares a risk class, and a policy turns the risk, the caller and the place into a decision.
// RiskClass is ordered, so rules read as thresholds and the order of the checks matters.
// the largest variant also trips every threshold rule, so it must be checked first.
// a denial is not an error, and its reason goes back to the model as a tool result.

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
        // Forbidden is denied first, then an authenticated read in a public place unless the config allows it.
        // a write at or above ExternalWrite without a write role is denied, and only then does confirm_from apply.
        todo!()
    }
}
