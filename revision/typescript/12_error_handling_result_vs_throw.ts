// throw is invisible in a function signature; a Result in the return type forces the caller to handle the failure.
// use Result for expected failures (bad input, not found, upstream down) and throw for defects (broken invariants, impossible branches).
// a Promise that rejects is a throw in disguise: catch at the adapter that owns the client and return a typed error value.
// give errors a stable _tag and structured fields; decide behavior by tag, never by matching message text.
// catch binds unknown, not Error: narrow before reading .message, and keep the original as cause.
// from halo: .agents/skills/coding-standards/references/errors.md says expected failures are values and defects throw.
// from halo: packages/api/src/generation/errors.ts defines tagged errors with unique tag strings and typed fields.
// from halo: packages/api/src/singular/SingularEvent.ts casesHandled throws for an impossible case.
// explain: why is a Result chain easier to test than code that throws in the middle of a function?
// explain: where does the last throw/catch boundary belong in an HTTP server, and what does it translate to?

export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

export type ParseError = { _tag: "ParseError"; message: string; cause: unknown };
export type NotFound = { _tag: "NotFound"; id: string };

export function tryCatch<T, E>(run: () => T, onError: (cause: unknown) => E): Result<T, E> {
  // run it; a throw becomes { ok: false, error: onError(cause) }. do not swallow non-throws.
  throw new Error("not implemented");
}

export async function tryPromise<T, E>(run: () => Promise<T>, onError: (cause: unknown) => E): Promise<Result<T, E>> {
  // the async twin: a rejection becomes an error value, a fulfilled value becomes ok. the returned promise never rejects.
  throw new Error("not implemented");
}

export function andThen<T, U, E, F>(result: Result<T, E>, next: (value: T) => Result<U, F>): Result<U, E | F> {
  // run next only on ok; an error short-circuits unchanged. the error type is the union of both.
  throw new Error("not implemented");
}

export function collectAll<T, E>(results: Array<Result<T, E>>): Result<T[], E> {
  // all ok -> the values in order; otherwise the FIRST error. empty input is ok with [].
  throw new Error("not implemented");
}

export function parseJson(text: string): Result<unknown, ParseError> {
  // JSON.parse inside tryCatch; keep the original exception as cause and put a stable message in message.
  throw new Error("not implemented");
}
