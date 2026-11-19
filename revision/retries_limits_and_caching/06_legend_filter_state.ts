// series visibility is a set of keys; never leave it empty. filter state becomes an
// order-insensitive fetch key so equal selections share a request. updates never mutate.

// 1. onLegendClick
// return a new Set; never mutate visible. checks:
// clicked not in all -> unchanged copy. clicked is the only visible key -> unchanged copy.
// clicked visible among several -> {clicked} (isolate). clicked hidden -> copy plus clicked.
// ex: ({a, b, c}, [a, b, c], b) -> {b},  ({b}, [a, b, c], a) -> {a, b}
//     ({b}, b) -> {b},  ({b}, z) -> {b}
export function onLegendClick(visible: ReadonlySet<string>, all: readonly string[], clicked: string): Set<string> {
  throw new Error("not implemented");
}

// 2. filterKey
// every filter list empty or undefined -> null (no request). else one string containing scope and
// every non-empty filter, filter names sorted and each filter's values sorted. separators are your
// choice, but different scopes/selections must give different strings. value order must not matter;
// an empty list equals an absent filter. don't mutate the lists.
// ex: ("tab", { region: ["b", "a"] }) === ("tab", { region: ["a", "b"] })
//     ("tab", { region: ["a"], year: [] }) === ("tab", { region: ["a"] })
//     ("tab", { region: [], year: undefined }) -> null
export function filterKey(scope: string, filters: Record<string, readonly string[] | undefined>): string | null {
  throw new Error("not implemented");
}

// 3. visibleSeries
// return a new array of series whose key is in visible, original order. empty visible -> [].
// don't mutate input.
// ex: keys [a, b, c], visible {c, a} -> [a, c]
// O(n) with Set membership, not a nested search.
export function visibleSeries<T extends { key: string }>(series: readonly T[], visible: ReadonlySet<string>): T[] {
  throw new Error("not implemented");
}
