// React state is a value you replace, never mutate: a new reference tells React something changed.
// a reducer (state, action) -> state is a pure function, so it is testable without React; useReducer is a thin shell around it.
// store the minimum (items, filter, selectedId); derive the rest (visible items, counts, the selected item) at read time.
// a stored copy of a derived value goes stale the moment the source changes; a derived value cannot.
// structural sharing: update only the path that changed and reuse every other reference, so memo checks stay cheap.
// returning the same state object for a no-op action lets React skip the re-render.
// from halo: apps/hq/src/components/templates/use-template-editor.ts exports templateEditorTransition(state, event), a pure reducer.
// from halo: apps/native/features/library/lib/local-modal.ts is another reducer, with local-modal.test.ts testing it without React.
// explain: two sibling components both need the filter. where does the state live, and what do you pass down?
// explain: why does storing visibleItems in state create a bug when an item is removed?

export type Item = { id: number; text: string; done: boolean };
export type Filter = "all" | "open" | "done";
export type State = { items: readonly Item[]; filter: Filter; selectedId: number | null; nextId: number };

export type Action =
  | { type: "add"; text: string }
  | { type: "toggle"; id: number }
  | { type: "remove"; id: number }
  | { type: "setFilter"; filter: Filter }
  | { type: "select"; id: number | null };

export function reducer(state: State, action: Action): State {
  // pure and immutable. add ignores blank text; toggle/remove of an unknown id return the SAME state object;
  // removing the selected item clears selectedId; select of an unknown id is a no-op. switch exhaustively.
  throw new Error("not implemented");
}

export function selectVisible(state: State): readonly Item[] {
  // items matching the filter, in order. derived on demand, never stored.
  throw new Error("not implemented");
}

export function selectSummary(state: State): { open: number; done: number; selected: Item | null } {
  // counts and the selected item looked up from state.items, so it can never be stale.
  throw new Error("not implemented");
}

export function setAt<T>(list: readonly T[], index: number, update: (item: T) => T): readonly T[] {
  // a new array with only index replaced; if update returns the identical value, return the original list.
  // every untouched element keeps its reference. out-of-range index returns the original list.
  throw new Error("not implemented");
}

export function replay(initial: State, actions: readonly Action[]): State {
  // fold the actions through the reducer. the same actions from the same state always give an equal state.
  throw new Error("not implemented");
}
