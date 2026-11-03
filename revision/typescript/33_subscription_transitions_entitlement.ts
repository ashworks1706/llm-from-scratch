// store events do not set access directly. they drive a transition table: (current status, event) -> action.
// access is then derived: entitled means status allows it AND expiresAt is in the future at read time.
// cancelled keeps access until expiry. expired only comes back through a renewal. unexpected pairs are noop, not errors.
// derive entitlement from stored facts plus a clock argument, so it is pure and testable.
//
// from halo: packages/api/src/subscriptions/transitions.ts (resolveTransition, LIVE/CANCELLED/EXPIRED tables) and
// subscriptions/entitlement.ts (isEntitledStatus, isSubscriptionEntitledNow, buildEntitlementState: no_subscription,
// active_subscription, expired_subscription). reproduce the table and the derivation on toy types.
//
// explain: why is a cancelled subscription still entitled, and what ends it?
// explain: why take now as a parameter instead of calling Date.now() inside?

type Status = "trialing" | "active" | "grace_period" | "billing_retry" | "cancelled" | "expired";
type Event = "renewal" | "cancellation" | "expiration" | "revocation" | "refund" | "payment_failed" | "payment_recovered";
type Action = "set_active" | "set_cancelled" | "set_expired" | "set_billing_retry" | "clear_and_expire" | "clear_and_revoke" | "reverse_and_expire" | "noop";
type Entitlement =
  | { _tag: "no_subscription" }
  | { _tag: "active_subscription"; status: Exclude<Status, "expired">; expiresAt: Date }
  | { _tag: "expired_subscription"; status: "expired" | "cancelled"; expiresAt: Date | null };

export function resolveTransition(from: Status, event: Event): Action {
  // live statuses (trialing, active, grace_period, billing_retry): renewal and payment_recovered re-activate, cancellation
  // marks cancelled, payment_failed moves to billing_retry, expiration clears credits and expires, revocation clears and
  // revokes, refund reverses credits and expires. cancelled: renewal re-activates, expiration clears and expires,
  // refund reverses and expires, the rest are noop. expired: only renewal acts.
  throw new Error("not implemented");
}

export function entitledNow(status: Status, expiresAt: Date | null, now: Date): boolean {
  // false when expiresAt is null, false when status is expired, otherwise expiresAt > now (strictly).
  throw new Error("not implemented");
}

export function buildEntitlement(sub: { status: Status; expiresAt: Date | null } | null, now: Date): Entitlement {
  // no row -> no_subscription. entitled now -> active_subscription. otherwise expired_subscription.
  // an expired view of a cancelled row keeps status cancelled; every other expired view says expired.
  throw new Error("not implemented");
}
