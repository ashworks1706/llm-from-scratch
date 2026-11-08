// Chart series visibility is a set of keys held in state, and the chart renders only what the set
// contains. Clicking a visible series isolates it, clicking a hidden series shows it again, and the
// set is never left empty. Filter state becomes a fetch key that ignores selection order, so equal
// selections share one request. State updates return new objects and never mutate the previous set.

// Problem 1: Legend click.
//
// visible is the set of series keys currently shown, all is the list of every series key, and
// clicked is the key of the legend entry the user clicked. Return a new Set with the next visible
// keys; never mutate the set passed in. If clicked is not in all, ignore it and return an
// unchanged copy. If clicked is the only visible key, return an unchanged copy so the chart is
// never empty. If clicked is visible among several visible keys, isolate it and return a set with
// just that key. If clicked is hidden, return a copy with clicked added.
//
// Example: visible {a, b, c}, all [a, b, c] and clicked b gives {b}.
// Example: visible {b}, all [a, b, c] and clicked a gives {a, b}.
// Example: visible {b} and clicked b gives {b}, and clicked z gives {b}.
export function onLegendClick(visible: ReadonlySet<string>, all: readonly string[], clicked: string): Set<string> {
  // if clicked is the only visible key, return an unchanged copy so the chart is never empty.
  // a visible click among several visible keys isolates it, a hidden click adds it back, and unknown keys are ignored.
  throw new Error("not implemented");
}

// Problem 2: Filter fetch key.
//
// scope is a string naming the screen or dataset, and filters maps a filter name to a list of
// selected values, or to undefined when that filter was never set. Return null when every filter
// list is empty or undefined, because no request is needed. Otherwise return one string that
// identifies the request: it includes scope and every non empty filter, with the filter names
// sorted and the values of each filter sorted. The choice of separators is yours, but different
// scopes or different selections must give different strings. The order of values must not
// matter, and a filter with an empty list must give the same key as a filter that is absent.
// Do not mutate the lists passed in.
//
// Example: filterKey("tab", { region: ["b", "a"] }) and filterKey("tab", { region: ["a", "b"] })
// give the same string.
// Example: filterKey("tab", { region: ["a"], year: [] }) equals filterKey("tab", { region: ["a"] }).
// Example: filterKey("tab", { region: [], year: undefined }) gives null.
export function filterKey(scope: string, filters: Record<string, readonly string[] | undefined>): string | null {
  // return null when every filter list is empty so no request is made, otherwise a string with names and values sorted.
  // ["b", "a"] and ["a", "b"] must give the same key, and an empty list must equal an absent one.
  throw new Error("not implemented");
}

// Problem 3: Visible series.
//
// series is an array of objects that each have a string key, and visible is the set of keys to
// show. Return a new array with the series whose key is in visible, in the same order as in
// series. An empty visible set gives an empty array. Do not mutate the input.
//
// Example: series with keys a, b, c and visible {c, a} gives the series a then c.
//
// Aim for O(n) using Set membership, not a nested search.
export function visibleSeries<T extends { key: string }>(series: readonly T[], visible: ReadonlySet<string>): T[] {
  // return the series whose key is in visible, keeping the original order, in O(n) time.
  throw new Error("not implemented");
}
