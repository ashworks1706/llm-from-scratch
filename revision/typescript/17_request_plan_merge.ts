// A screen showing many indicators groups them into one request per endpoint instead of one request
// per indicator. Each request carries the deduplicated union of codes and years its indicators
// need. Responses merge into one row per place, and when several periods exist for one field the
// latest period wins. A row exists only for places present in a response, and a missing value stays
// missing rather than becoming zero.

export interface IndicatorSpec { key: string; endpoint: string; code: string; years: string[] }
export interface Observation { code: string; place: string; startYear: number; value: number | null }
export interface Bucket { codes: Set<string>; years: Set<string>; keysByCode: Map<string, string[]> }
export type Plan = Record<string, Bucket>;

// Problem 1: Build the request plan.
//
// selected is the list of indicator keys the screen shows and specs maps an indicator key to its
// spec, which names the endpoint, the code to request and the years needed. Return a Plan, an
// object keyed by endpoint with one Bucket per endpoint that at least one selected indicator uses.
// A bucket holds codes, the set of codes to request; years, the set of all years needed by its
// indicators; and keysByCode, a Map from each code to the list of selected indicator keys that use
// it, in selection order. Skip selected keys that have no spec. If two indicators share a code,
// both keys appear in keysByCode for that code, while the code is in codes only once. Years are
// deduplicated across all indicators of the endpoint. An empty selection gives {}.
//
// Example: selected [gdp, pop, nope], with gdp on endpoint /wb with code NY.GDP and years 2020 and
// 2021, pop on /wb with code SP.POP and year 2021, and no spec for nope, gives one bucket for /wb
// with codes {NY.GDP, SP.POP}, years {2020, 2021}, and keysByCode NY.GDP to [gdp], SP.POP to [pop].
export function buildPlan(selected: readonly string[], specs: Record<string, IndicatorSpec>): Plan {
  // return one bucket per endpoint, skipping selected keys that have no spec.
  // two indicators sharing a code add both keys to keysByCode but the code to codes only once.
  throw new Error("not implemented");
}

// Problem 2: Merge responses into rows.
//
// plan is the result of buildPlan and responses maps an endpoint to the list of observations the
// server returned for it. Return an object from place to a record of indicator key to value. For
// each observation, look up its code in the keysByCode of that endpoint's bucket, and write the
// value under every key mapped to that code; skip observations whose endpoint or code is not in the
// plan. Process each response's observations in ascending startYear order so that a later period
// overwrites an earlier one for the same place and key. A place gets a row only if it appears in
// some observation, and a null value is stored as null, never converted to 0. Sort a copy so the
// response arrays passed in are not mutated.
//
// Example: with the plan from buildPlan above and responses { "/wb": [ { code: "NY.GDP", place:
// "Oslo", startYear: 2021, value: 5 }, { code: "NY.GDP", place: "Oslo", startYear: 2020, value: 3
// }, { code: "SP.POP", place: "Oslo", startYear: 2021, value: null } ] } the result is
// { Oslo: { gdp: 5, pop: null } }, since 2021 overwrites 2020.
//
// Aim for O(n log n) in the number of observations because of the sort.
export function mergeRows(plan: Plan, responses: Record<string, Observation[]>): Record<string, Record<string, number | null>> {
  // return place to { indicatorKey: value }, writing every key mapped to the code of each observation.
  // sort each response by startYear ascending before writing so later periods overwrite earlier ones, O(n log n).
  throw new Error("not implemented");
}
