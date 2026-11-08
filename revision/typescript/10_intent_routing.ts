// A turn turns free text and attachments into a job packet, and the rules plan is computed first as
// the floor. Rules look only at what is attached, so planning needs no io and no model call. With
// no new media attached the newest completed result becomes the source, so a follow-up keeps
// editing it. Routing maps the operation and planned family to one endpoint plus a reason string,
// and an unknown family falls back.

type Turn = { status: "pending" | "processing" | "completed" | "failed"; kind: "image" | "video"; outputUrl: string | null };
type Request = { prompt: string; modality: "auto" | "image" | "video"; imageUris: string[]; videoUri?: string };

// Problem 1: Continuity from the previous turn.
//
// request is the new user request and turns is the conversation history, listed oldest first so the
// newest turn is the last element. If the request already has a video (videoUri is set) or at least
// one image (imageUris is not empty), return it unchanged, since the user attached new media. If it
// has neither, look for the newest turn whose status is completed and whose outputUrl is not null,
// and return a copy of the request with that outputUrl as its source: an image turn puts the url as
// the only entry of imageUris, and a video turn sets videoUri. A video turn is not eligible when
// request.modality is "image", so skip it and keep looking at older turns. If no turn is eligible,
// return the request unchanged. The request passed in is never mutated.
//
// Example: request { prompt: "make it red", modality: "auto", imageUris: [] } with turns
// [{ status: "completed", kind: "image", outputUrl: "u1" }, { status: "failed", kind: "image",
// outputUrl: null }] gives a request with imageUris ["u1"], since the failed turn is skipped.
// Example: a request that already has imageUris ["mine"] is returned with imageUris ["mine"].
export function withContinuity(request: Request, turns: Turn[]): Request {
  // with no clip and no images reuse the newest completed turn that has an outputUrl.
  // a video result must not be reused when modality is pinned to image, and the input is not mutated.
  throw new Error("not implemented");
}

// Problem 2: Route an image job to an endpoint.
//
// operation is text_to_image or image_edit. plannedFamily is the model family chosen by the rules
// planner, or null if none. endpoints maps a family name to its two endpoint urls, one for edits
// and one for text to image. fallbackFamily names the family to use when the planned one cannot be
// used; assume it is a key of endpoints, and throw an Error if it is not. Return an object with
// the endpoint url, the family actually used, and a reason string. When plannedFamily is not null
// and is a key of endpoints, use it: the endpoint is its edit url for image_edit or its
// textToImage url for text_to_image, and the reason says the planned family was used. Otherwise
// use fallbackFamily the same way, and make the reason say that the planned family was missing or
// unknown and the fallback was used.
//
// Example: endpoints { flux: { edit: "/flux/edit", textToImage: "/flux/t2i" } }, operation
// image_edit, plannedFamily "flux" gives endpoint "/flux/edit" and family "flux".
// Example: the same endpoints, operation text_to_image, plannedFamily "nope", fallbackFamily
// "flux" gives endpoint "/flux/t2i" and family "flux", with a reason mentioning the fallback.
export function routeImage(operation: "text_to_image" | "image_edit", plannedFamily: string | null, endpoints: Record<string, { edit: string; textToImage: string }>, fallbackFamily: string): { endpoint: string; family: string; reason: string } {
  // use the planned family only if it exists in endpoints, otherwise use fallbackFamily and say so in reason.
  throw new Error("not implemented");
}
