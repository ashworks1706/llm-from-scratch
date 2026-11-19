// group indicators into one request per endpoint with deduped codes and years, then merge responses
// into one row per place: latest period wins, missing values stay missing (not zero).

export interface IndicatorSpec { key: string; endpoint: string; code: string; years: string[] }
export interface Observation { code: string; place: string; startYear: number; value: number | null }
export interface Bucket { codes: Set<string>; years: Set<string>; keysByCode: Map<string, string[]> }
export type Plan = Record<string, Bucket>;

// 1. buildPlan
// return { endpoint: Bucket } for each endpoint used by at least one selected indicator.
// codes: codes to request (each once); years: all years needed, deduped across the endpoint;
// keysByCode: code -> selected keys using it, in selection order (shared code lists both keys).
// skip selected keys with no spec. [] -> {}.
// ex: [gdp, pop, nope], gdp: /wb NY.GDP [2020, 2021], pop: /wb SP.POP [2021], nope: no spec
//     -> { "/wb": codes {NY.GDP, SP.POP}, years {2020, 2021},
//          keysByCode NY.GDP -> [gdp], SP.POP -> [pop] }
export function buildPlan(selected: readonly string[], specs: Record<string, IndicatorSpec>): Plan {
  throw new Error("not implemented");
}

// 2. mergeRows
// return { place: { indicatorKey: value } }. for each observation, look up its code in that
// endpoint's keysByCode and write value under every mapped key; skip unknown endpoints or codes.
// process each response in ascending startYear so later periods overwrite earlier (same place+key).
// a place gets a row only if it appears in some observation. null stays null, never 0.
// sort a copy; don't mutate the response arrays.
// ex (code, place, startYear, value): plan above,
//     { "/wb": [(NY.GDP, Oslo, 2021, 5), (NY.GDP, Oslo, 2020, 3), (SP.POP, Oslo, 2021, null)] }
//     -> { Oslo: { gdp: 5, pop: null } }
// O(n log n) from the sort.
export function mergeRows(plan: Plan, responses: Record<string, Observation[]>): Record<string, Record<string, number | null>> {
  throw new Error("not implemented");
}
