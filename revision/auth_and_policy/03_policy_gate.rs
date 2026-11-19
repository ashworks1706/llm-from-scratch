// Background: a policy turns (risk, caller, place) into a decision. RiskClass is ordered so rules
// read as thresholds; check order matters. a denial isn't an error; its reason goes to the model.

// least -> most dangerous, ordered (risk >= RiskClass::ExternalWrite works). PrepareWrite drafts
// without applying, ExternalWrite changes outside state, Destructive is irreversible,
// Forbidden never runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RiskClass { ReadPublic, ReadAuthenticated, PrepareWrite, ExternalWrite, Destructive, Forbidden }

// Confirm = hold until a human approves.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Decision { Allow, Deny(String), Confirm }

// private: true in a private place, false in public.
struct Ctx { roles: Vec<String>, private: bool }

// write_roles empty = nobody writes. allow_authenticated_reads: ReadAuthenticated ok in public.
// confirm_from: lowest risk needing human confirmation.
struct RiskPolicy { write_roles: Vec<String>, allow_authenticated_reads: bool, confirm_from: RiskClass }

impl RiskPolicy {
    // 1. may_write
    // true if the caller holds any role in write_roles. empty write_roles or no roles -> false.
    // ex: write_roles ["officer"]: ["member", "officer"] -> true, ["member"] -> false
    fn may_write(&self, ctx: &Ctx) -> bool {
        todo!()
    }

    // 2. authorize
    // first match wins, in this order:
    //   1) Forbidden -> Deny
    //   2) ReadAuthenticated in public (!private) -> Deny unless allow_authenticated_reads
    //   3) risk >= ExternalWrite and !may_write -> Deny
    //   4) risk >= confirm_from -> Confirm
    //   5) Allow
    // Deny text says why (the model reads it).
    // ex: policy write_roles ["officer"], auth reads false, confirm_from ExternalWrite:
    //     private officer: ReadPublic -> Allow, Destructive -> Confirm, Forbidden -> Deny
    //     private, no roles: ExternalWrite -> Deny.  public caller: ReadAuthenticated -> Deny
    //     confirm_from Destructive, private officer: ExternalWrite -> Allow
    fn authorize(&self, ctx: &Ctx, risk: RiskClass) -> Decision {
        todo!()
    }
}
