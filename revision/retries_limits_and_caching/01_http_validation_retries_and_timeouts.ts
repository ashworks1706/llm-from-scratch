// apis fail by timeout, connection loss, 429, bad json. validate at the boundary, retry only safe
// operations with bounded attempts, and cancel slow fetches with AbortController.

// 1. fetchJsonWithTimeout
// fetch url, return the parsed json body as unknown (shape not checked).
// reject if: not done within timeoutMs (abort via AbortController), status not 2xx, or bad json.
// clear the timer once the request settles. no retries: one call = one request.
export async function fetchJsonWithTimeout(
  url: string,
  timeoutMs: number
): Promise<unknown> {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
// needs a url to hit; start a local server first, e.g. in another shell: python3 -m http.server 8000
// assert.deepEqual(await fetchJsonWithTimeout("http://localhost:8000/data.json", 1000), { a: 1 }); // put {"a":1} in data.json first
// await assert.rejects(fetchJsonWithTimeout("http://localhost:8000/", 1000)); // html, not json
// await assert.rejects(fetchJsonWithTimeout("http://localhost:1/", 500));      // connection refused
