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

struct Ctx { roles: Vec<String>, private: bool }

// write_roles empty = nobody writes. allow_authenticated_reads: ReadAuthenticated ok in public.
// confirm_from: lowest risk needing human confirmation.
struct RiskPolicy { write_roles: Vec<String>, allow_authenticated_reads: bool, confirm_from: RiskClass }

impl RiskPolicy {
    // 1. may_write
    // true if the caller holds any role in write_roles. empty write_roles or no roles -> false.
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
    fn authorize(&self, ctx: &Ctx, risk: RiskClass) -> Decision {
        todo!()
    }
}

// ---- try it ----
fn main() {
    let policy = RiskPolicy { write_roles: vec!["officer".to_string()], allow_authenticated_reads: false, confirm_from: RiskClass::ExternalWrite };
    let officer = Ctx { roles: vec!["member".to_string(), "officer".to_string()], private: true };
    let nobody = Ctx { roles: vec![], private: true };
    let public = Ctx { roles: vec![], private: false };
    // assert!(policy.may_write(&officer));
    // assert!(!policy.may_write(&nobody));
    // assert_eq!(policy.authorize(&officer, RiskClass::ReadPublic), Decision::Allow);
    // assert_eq!(policy.authorize(&officer, RiskClass::Destructive), Decision::Confirm);
    // assert!(matches!(policy.authorize(&officer, RiskClass::Forbidden), Decision::Deny(_)));
    // assert!(matches!(policy.authorize(&nobody, RiskClass::ExternalWrite), Decision::Deny(_)));
    // assert!(matches!(policy.authorize(&public, RiskClass::ReadAuthenticated), Decision::Deny(_)));
    // let lax = RiskPolicy { confirm_from: RiskClass::Destructive, ..policy };
    // assert_eq!(lax.authorize(&officer, RiskClass::ExternalWrite), Decision::Allow);
}
