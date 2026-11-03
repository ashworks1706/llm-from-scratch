// a turn turns free text and attachments into a job packet, and the rules plan is computed first as the floor.
// rules look only at what is attached, so planning needs no io and no model call.
// with no new media attached the newest completed result becomes the source, so a follow-up keeps editing it.
// routing maps the operation and planned family to one endpoint plus a reason string, and an unknown family falls back.

type Turn = { status: "pending" | "processing" | "completed" | "failed"; kind: "image" | "video"; outputUrl: string | null };
type Request = { prompt: string; modality: "auto" | "image" | "video"; imageUris: string[]; videoUri?: string };

export function withContinuity(request: Request, turns: Turn[]): Request {
  // with no clip and no images reuse the newest completed turn that has an outputUrl.
  // a video result must not be reused when modality is pinned to image, and the input is not mutated.
  throw new Error("not implemented");
}

export function routeImage(operation: "text_to_image" | "image_edit", plannedFamily: string | null, endpoints: Record<string, { edit: string; textToImage: string }>, fallbackFamily: string): { endpoint: string; family: string; reason: string } {
  // use the planned family only if it exists in endpoints, otherwise use fallbackFamily and say so in reason.
  throw new Error("not implemented");
}
