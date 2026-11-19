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
// ex: ({ prompt: "make it red", modality: "auto", imageUris: [] },
//      [{ completed, image, "u1" }, { failed, image, null }]) -> imageUris ["u1"] (failed skipped)
//     request with imageUris ["mine"] -> imageUris ["mine"]
export function withContinuity(request: Request, turns: Turn[]): Request {
  throw new Error("not implemented");
}

// 2. routeImage
// plannedFamily non-null and a key of endpoints -> use it, reason says planned family was used.
// else use fallbackFamily, reason says planned was missing/unknown and the fallback was used.
// fallbackFamily not in endpoints -> throw Error.
// endpoint = family's edit url for image_edit, textToImage url for text_to_image.
// ex: endpoints { flux: { edit: "/flux/edit", textToImage: "/flux/t2i" } }
//     (image_edit, "flux") -> { endpoint: "/flux/edit", family: "flux", ... }
//     (text_to_image, "nope", fallback "flux")
//       -> { endpoint: "/flux/t2i", family: "flux", reason mentions fallback }
export function routeImage(operation: "text_to_image" | "image_edit", plannedFamily: string | null, endpoints: Record<string, { edit: string; textToImage: string }>, fallbackFamily: string): { endpoint: string; family: string; reason: string } {
  throw new Error("not implemented");
}
