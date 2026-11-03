// retry only what is safe to repeat. an error says two separate things: can I retry, and might it have already happened.
// timeouts and lost connections are retryable AND may have applied. a 409 conflict after one of those means the write landed.
// so after retries end, a conflict is reported as the earlier may-have-applied error, not as a plain conflict.
// idempotency keys make the repeat safe, but the caller still needs the truth about the last unknown.
//
// from halo: packages/api/src/generation/GenerationBilling.ts (withRevenueCatRetry: exponential 250 ms, 3 retries, only while
// isRetryable; mayHaveAppliedError wins over Conflict or Locked) and generation/provider-job-state.ts (bounded: exponential
// 100 ms, 3 retries, 10 s timeout). reproduce the retry loop with an injected sleep so it tests fast.
//
// explain: why surface the earlier unknown instead of the final conflict?
// explain: why exponential backoff with a retry cap, and what does jitter add when many workers retry together?

type Failure = { reason: string; isRetryable: boolean; mayHaveApplied: boolean };

export async function retryWithBackoff<T>(
  op: () => Promise<T>,
  classify: (error: unknown) => Failure | null,
  options: { baseMs: number; maxRetries: number; sleep: (ms: number) => Promise<void> },
): Promise<T> {
  // run op. on error: classify; null (unknown error) rethrows immediately, a non-retryable failure rethrows.
  // delay before retry k (k from 1) is baseMs * 2^(k-1). at most maxRetries retries, so maxRetries + 1 calls total.
  // remember the first failure with mayHaveApplied true. when finally giving up, if the last error has reason
  // "Conflict" or "Locked" and a may-have-applied failure was seen earlier, throw that earlier one instead.
  throw new Error("not implemented");
}

export function delays(baseMs: number, maxRetries: number): number[] {
  // the schedule alone, for assertions: [250, 500, 1000] for base 250 and 3 retries.
  throw new Error("not implemented");
}
