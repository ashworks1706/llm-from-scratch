// plan from attachments with rules only (no io, no model call). with no new media, the newest
// completed result becomes the source so follow-ups keep editing it. routing maps
// operation + planned family to one endpoint and a reason; unknown families fall back.

type Turn = { status: "pending" | "processing" | "completed" | "failed"; kind: "image" | "video"; outputUrl: string | null };
type Request = { prompt: string; modality: "auto" | "image" | "video"; imageUris: string[]; videoUri?: string };

// 1. withContinuity
// turns are oldest first. request has videoUri or non-empty imageUris -> return it unchanged.
// else find the newest turn with status completed and outputUrl not null, return a copy with it as
// source: image turn -> imageUris = [url], video turn -> videoUri = url.
// if request.modality is "image", skip video turns and keep looking older.
// no eligible turn -> request unchanged. never mutate the request.
export function withContinuity(request: Request, turns: Turn[]): Request {
  throw new Error("not implemented");
}

// 2. routeImage
// plannedFamily non-null and a key of endpoints -> use it, reason says planned family was used.
// else use fallbackFamily, reason says planned was missing/unknown and the fallback was used.
// fallbackFamily not in endpoints -> throw Error.
// endpoint = family's edit url for image_edit, textToImage url for text_to_image.
export function routeImage(operation: "text_to_image" | "image_edit", plannedFamily: string | null, endpoints: Record<string, { edit: string; textToImage: string }>, fallbackFamily: string): { endpoint: string; family: string; reason: string } {
  throw new Error("not implemented");
}

// ---- try it ----
import assert from "node:assert/strict";
const req: Request = { prompt: "make it red", modality: "auto", imageUris: [] };
const turns: Turn[] = [
  { status: "completed", kind: "image", outputUrl: "u1" },
  { status: "failed", kind: "image", outputUrl: null },
];
const endpoints = { flux: { edit: "/flux/edit", textToImage: "/flux/t2i" } };
// assert.deepEqual(withContinuity(req, turns).imageUris, ["u1"]);
// assert.deepEqual(withContinuity({ ...req, imageUris: ["mine"] }, turns).imageUris, ["mine"]);
// { const r = routeImage("image_edit", "flux", endpoints, "flux"); assert.deepEqual([r.endpoint, r.family], ["/flux/edit", "flux"]); } // reason: planned used
// { const r = routeImage("text_to_image", "nope", endpoints, "flux"); assert.deepEqual([r.endpoint, r.family], ["/flux/t2i", "flux"]); } // reason: fallback
