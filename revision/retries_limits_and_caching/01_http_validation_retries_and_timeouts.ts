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
// needs a url to hit; start a local server first, e.g. in another shell: python3 -m http.server 8000
// fetchJsonWithTimeout("http://localhost:8000/data.json", 1000).then(console.log);   // { a: 1 } (put {"a":1} in data.json first)
// fetchJsonWithTimeout("http://localhost:8000/", 1000).then(console.log, (e) => console.log("rejected:", e.message)); // rejects (html, not json)
// fetchJsonWithTimeout("http://localhost:1/", 500).then(console.log, (e) => console.log("rejected:", e.message));      // rejects (connection refused)
