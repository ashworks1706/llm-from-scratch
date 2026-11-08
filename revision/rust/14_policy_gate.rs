// Background: every tool declares a risk class, and a policy turns the risk, the caller and the
// place into a decision. RiskClass is ordered, so rules read as thresholds and the order of the
// checks matters. The largest variant also trips every threshold rule, so it must be checked
// first. A denial is not an error, and its reason goes back to the model as a tool result.

// How dangerous a tool is, from least to most. The variants are ordered in this order, so
// comparisons such as risk >= RiskClass::ExternalWrite work. ReadPublic reads public data.
// ReadAuthenticated reads data that needs a signed in user. PrepareWrite drafts a change without
// applying it. ExternalWrite changes something outside the system. Destructive deletes or cannot be
// undone. Forbidden must never run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RiskClass { ReadPublic, ReadAuthenticated, PrepareWrite, ExternalWrite, Destructive, Forbidden }

// The outcome of a policy check. Allow means run the tool now. Deny carries a reason text and
// means do not run it. Confirm means hold the action until a human approves it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Decision { Allow, Deny(String), Confirm }

// Who is calling and from where. roles lists the roles the caller holds. private is true when the
// call happens in a private place and false in a public one.
struct Ctx { roles: Vec<String>, private: bool }

// The rules. write_roles lists the roles that may perform writes, and an empty list means nobody.
// allow_authenticated_reads says whether ReadAuthenticated tools may run in a public place.
// confirm_from is the lowest risk class that needs human confirmation.
struct RiskPolicy { write_roles: Vec<String>, allow_authenticated_reads: bool, confirm_from: RiskClass }

// Problem: RiskPolicy.
//
// RiskPolicy decides what happens when a caller wants to run a tool of a given risk class.
impl RiskPolicy {
    // Problem: may_write.
    //
    // Return true when the caller holds at least one role that appears in write_roles. If
    // write_roles is empty, return false for everyone. If the caller has no roles, return false.
    //
    // Example: with write_roles ["officer"], a caller with roles ["member", "officer"] gives true
    // and a caller with roles ["member"] gives false.
    fn may_write(&self, ctx: &Ctx) -> bool {
        todo!()
    }

    // Problem: authorize.
    //
    // Given the caller context and a risk class, return a Decision. Apply these checks in this
    // order and return at the first one that matches. One: Forbidden is denied, with a reason.
    // Two: a ReadAuthenticated tool in a public place (private is false) is denied unless
    // allow_authenticated_reads is true. Three: a risk of ExternalWrite or higher is denied when
    // may_write is false. Four: only now apply confirm_from, and return Confirm when risk is at or
    // above it. Five: return Allow. The Deny text should say why, because the model reads it.
    //
    // Example: with write_roles ["officer"], allow_authenticated_reads false, confirm_from
    // ExternalWrite, and a private caller with roles ["officer"]: ReadPublic gives Allow,
    // Destructive gives Confirm and Forbidden gives Deny.
    // Example: the same policy with a private caller who has no roles gives Deny for ExternalWrite.
    // Example: the same policy with a public caller gives Deny for ReadAuthenticated.
    // Example: with confirm_from Destructive, a private officer asking for ExternalWrite gets Allow.
    fn authorize(&self, ctx: &Ctx, risk: RiskClass) -> Decision {
        todo!()
    }
}
