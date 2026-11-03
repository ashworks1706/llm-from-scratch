// APIs can fail by timeout, connection loss, 429 rate limit or malformed JSON.
// validate incoming data at the boundary; type annotations alone cannot help.
// retry only operations that are safe to repeat, and use bounded attempts.
// an AbortController can cancel a fetch that exceeds a deadline.

export async function fetchJsonWithTimeout(
  url: string,
  timeoutMs: number
): Promise<unknown> {
  // return parsed JSON; abort on timeout and reject non-2xx status codes.
  throw new Error("not implemented");
}
