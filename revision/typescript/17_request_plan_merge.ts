// a screen showing many indicators groups them into one request per endpoint instead of one request per indicator.
// each request carries the deduplicated union of codes and years its indicators need.
// responses merge into one row per place, and when several periods exist for one field the latest period wins.
// a row exists only for places present in a response, and a missing value stays missing rather than becoming zero.

export interface IndicatorSpec { key: string; endpoint: string; code: string; years: string[] }
export interface Observation { code: string; place: string; startYear: number; value: number | null }
export interface Bucket { codes: Set<string>; years: Set<string>; keysByCode: Map<string, string[]> }
export type Plan = Record<string, Bucket>;

export function buildPlan(selected: readonly string[], specs: Record<string, IndicatorSpec>): Plan {
  // return one bucket per endpoint, skipping selected keys that have no spec.
  // two indicators sharing a code add both keys to keysByCode but the code to codes only once.
  throw new Error("not implemented");
}

export function mergeRows(plan: Plan, responses: Record<string, Observation[]>): Record<string, Record<string, number | null>> {
  // return place to { indicatorKey: value }, writing every key mapped to the code of each observation.
  // sort each response by startYear ascending before writing so later periods overwrite earlier ones, O(n log n).
  throw new Error("not implemented");
}
