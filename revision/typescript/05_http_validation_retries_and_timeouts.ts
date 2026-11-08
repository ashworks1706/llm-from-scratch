// APIs can fail by timeout, connection loss, 429 rate limit or malformed JSON. Validate incoming
// data at the boundary, because type annotations alone cannot help. Retry only operations that are
// safe to repeat, and use bounded attempts. An AbortController can cancel a fetch that exceeds a
// deadline.

// Problem: Fetch JSON with a timeout.
//
// url is the address to request with fetch and timeoutMs is the deadline in milliseconds. Start the
// request and return the parsed JSON body as unknown, since its shape is not checked here. Abort
// the request with an AbortController if it has not completed within timeoutMs, and make the
// returned promise reject in that case. Also reject when the response status is not in the 2xx
// range, and reject when the body is not valid JSON. Clear the timer once the request settles so it
// does not keep running. Do not retry; one call makes one request.
//
// Example: a server that answers 200 with the body {"a":1} gives { a: 1 }.
// Example: a server that answers 500 gives a rejected promise.
// Example: a server that takes longer than timeoutMs to answer gives a rejected promise shortly
// after timeoutMs has passed.
export async function fetchJsonWithTimeout(
  url: string,
  timeoutMs: number
): Promise<unknown> {
  // return parsed JSON; abort on timeout and reject non-2xx status codes.
  throw new Error("not implemented");
}
