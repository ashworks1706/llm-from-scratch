// apis fail by timeout, connection loss, 429, bad json. validate at the boundary, retry only safe
// operations with bounded attempts, and cancel slow fetches with AbortController.

// 1. fetchJsonWithTimeout
// fetch url, return the parsed json body as unknown (shape not checked).
// reject if: not done within timeoutMs (abort via AbortController), status not 2xx, or bad json.
// clear the timer once the request settles. no retries: one call = one request.
// ex: 200 {"a":1} -> { a: 1 },  500 -> rejects,  slower than timeoutMs -> rejects shortly after
export async function fetchJsonWithTimeout(
  url: string,
  timeoutMs: number
): Promise<unknown> {
  throw new Error("not implemented");
}
