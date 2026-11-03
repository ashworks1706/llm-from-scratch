// chart series visibility is a set of keys held in state, and the chart renders only what the set contains.
// clicking a visible series isolates it, clicking a hidden series shows it again, and the set is never left empty.
// filter state becomes a fetch key that ignores selection order, so equal selections share one request.
// state updates return new objects and never mutate the previous set.

export function onLegendClick(visible: ReadonlySet<string>, all: readonly string[], clicked: string): Set<string> {
  // if clicked is the only visible key, return an unchanged copy so the chart is never empty.
  // a visible click among several visible keys isolates it, a hidden click adds it back, and unknown keys are ignored.
  throw new Error("not implemented");
}

export function filterKey(scope: string, filters: Record<string, readonly string[] | undefined>): string | null {
  // return null when every filter list is empty so no request is made, otherwise a string with names and values sorted.
  // ["b", "a"] and ["a", "b"] must give the same key, and an empty list must equal an absent one.
  throw new Error("not implemented");
}

export function visibleSeries<T extends { key: string }>(series: readonly T[], visible: ReadonlySet<string>): T[] {
  // return the series whose key is in visible, keeping the original order, in O(n) time.
  throw new Error("not implemented");
}
