// experiments are declared in one registry: key, variants, a fallback variant, a typed payload and where they may run.
// every read has a safe answer. unknown key, bad variant or an undecodable payload returns the fallback, never throws.
// the registry validates itself so a bad declaration fails in a test, not in production.
// assignments travel as a header "key:variant,key:variant" and are decoded defensively on the other side.
//
// from halo: packages/experiments/src/registry.ts (defineExperiment, createExperimentRegistry, totalPayloadDecoder,
// decodeFeatureFlagsHeader, validateExperimentRegistry) and packages/experiments/src/entries.ts (the declared experiments).
// reproduce validation, header decode and the total payload decoder with no schema library.
//
// explain: why must a flag read never throw on the request path, and what is the cost of returning the fallback?
// explain: why can a device-bucketed experiment not run on the server?

type Entry = { key: string; variants: string[]; fallbackVariant: string; bucketing: "device" | "user" | "global"; surfaces: Array<"native" | "server"> };

export function validateRegistry(entries: Entry[]): string[] {
  // return one message per problem, empty array when valid. rules: key is kebab-case [a-z0-9-], no duplicate keys,
  // at least one variant, no duplicate variants, variants contain no whitespace, ":" or ",", fallbackVariant is declared,
  // device bucketing cannot list the server surface, global bucketing lists only the server surface.
  throw new Error("not implemented");
}

export function decodeHeader(raw: string | null | undefined, entries: Entry[]): Record<string, string> {
  // split on ",", trim, skip empty tokens. split each token at the FIRST ":".
  // drop tokens with no key, no variant, an unknown key or a variant not declared for that key. last write wins on repeats.
  throw new Error("not implemented");
}

export function totalDecoder<T>(decode: (raw: unknown) => T | null, fallback: T): (raw: unknown) => T {
  // returns decode(raw) when it is non-null, otherwise fallback. also fallback when decode throws.
  throw new Error("not implemented");
}
