// the llm never sees a url. it sees handles (upload_1, video_1, result_prev) and may only answer with handles.
// code resolves handles back to urls. an unknown handle is dropped and logged as an override, never guessed.
// this makes references checkable, saves tokens, and closes the door on injected urls.
// handles are stable for one turn: uploads first, then past results, the newest result is always result_prev.
//
// from halo: packages/api/src/planner/handles.ts (buildHandleTable, resolveHandles, unknownHandlesOverride).
// reproduce the table and the resolver on toy types.
//
// explain: why is a symbol table safer than letting the model echo urls back?
// explain: why alias result_N to result_prev for only the newest result?

type Upload = { uri: string; mediaType: "image" | "video" };
type PastTurn = { outputUrl: string | null; kind: "image" | "video" };
type Handle = { id: string; type: "image" | "video"; url: string; source: "upload" | "result" };
type Table = { media: Handle[]; aliases: Map<string, string> };

export function buildTable(uploads: Upload[], history: PastTurn[]): Table {
  // images count upload_1, upload_2...; videos count separately as video_1, video_2...
  // a past turn with outputUrl null adds nothing. the turn number is its 1-based index in history, null outputs included.
  // the newest result gets id result_prev and an alias result_<turn> -> result_prev; older ones keep result_<turn>.
  throw new Error("not implemented");
}

export function resolve(table: Table, inputs: string[]): { resolved: Handle[]; unknown: string[] } {
  // normalize each input with trim + lowercase, then follow the alias if there is one.
  // not found -> push the ORIGINAL string to unknown. two handles with the same url resolve once.
  // keep the order the model gave. O(n) with a Map.
  throw new Error("not implemented");
}
