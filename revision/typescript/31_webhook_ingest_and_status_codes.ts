// a webhook handler decides two things: what to tell the sender, and whether the event is durable before saying yes.
// the sender retries on 5xx and stops on 2xx, so the status code is a retry policy.
// retryable or unknown failure -> 5xx (redeliver). bad auth -> 401. bad payload -> 400. a known unprocessable event -> 200 (stop).
// dedupe is a durable inbox: insert keyed by the provider event id, on conflict do nothing. ack only after the insert.
//
// from halo: packages/api/src/webhooks/revenuecat.ts (revenueCatWebhookOutcome, onAuthorizationError, onInvalidPayload,
// onSchemaMismatch, onHttpError, revenueCatWebhookHandler returns 500 when the ledger write fails) and ADR
// docs/adr/0147-halo-owns-singular-purchase-feed.md (inbox dedupe on rc:<event id>, ack after the row is durable).
// reproduce the status mapping and the inbox on toy types.
//
// explain: why is 200 on a non-retryable upstream error right, and what would a 500 there cause?
// explain: why must the inbox insert happen before the 200, and what is lost if you ack first and crash?

type WebhookFailure =
  | { _tag: "AuthNotConfigured" }
  | { _tag: "AuthMissing" }
  | { _tag: "AuthInvalid" }
  | { _tag: "PayloadInvalid" }
  | { _tag: "SchemaMismatch" }
  | { _tag: "UpstreamHttp"; isRetryable: boolean }
  | { _tag: "Defect" };

export function statusFor(failure: WebhookFailure): 200 | 400 | 401 | 500 {
  // exhaustive switch, no default. misconfigured auth is our bug (500), not the caller's (401).
  // schema mismatch and defects are 500 so the sender redelivers after a fix. non-retryable upstream is 200.
  throw new Error("not implemented");
}

export class Inbox {
  private seen = new Set<string>();

  record(eventId: string): "recorded" | "duplicate" {
    // key is "rc:" + eventId. first insert returns recorded, any repeat returns duplicate. O(1).
    throw new Error("not implemented");
  }
}

export function handle(inbox: Inbox, eventId: string, writeLedger: (eventId: string) => void): { status: 200 | 500; outcome: string } {
  // duplicate -> 200 without calling writeLedger again. recorded -> call writeLedger, 200 on success.
  // writeLedger throws -> forget the inbox key (so the redelivery is not seen as a duplicate) and return 500.
  throw new Error("not implemented");
}
