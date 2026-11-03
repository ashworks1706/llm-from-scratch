// a turn turns free text plus attachments into a job packet. the rules plan is computed first and is the floor.
// rules look only at what is attached: a clip, 2+ images, 1 image, nothing. no io, no model call.
// with no new media attached, the newest completed result becomes the source, so "make it warmer" keeps editing.
// routing then maps (operation, planned family) to one endpoint plus a reason string. an unknown family falls back.
//
// from halo: packages/api/src/routing/turn-constraints.ts (deterministicPlan, deterministicImagePacket,
// deterministicVideoPacket) and routing/route-packet.ts (routeGeneration, routeImage, ROUTING_FALLBACK_IMAGE_MODEL).
// reproduce the rules on toy types, not the service.
//
// explain: why compute the rules plan before the llm call even when the llm usually wins?
// explain: why does a route carry a reason string, and who reads it later?

type Turn = { status: "pending" | "processing" | "completed" | "failed"; kind: "image" | "video"; outputUrl: string | null };
type Request = { prompt: string; modality: "auto" | "image" | "video"; imageUris: string[]; videoUri?: string };
type Operation = "text_to_image" | "image_edit" | "text_to_video" | "image_to_video" | "reference_to_video" | "video_edit";
type Packet = { kind: "image" | "video"; operation: Operation; prompt: string; imageUris: string[]; aspectRatio: string };

export function withContinuity(request: Request, turns: Turn[]): Request {
  // if the request has no clip and no images, reuse the newest completed turn that has an outputUrl.
  // a video result becomes videoUri, an image result becomes imageUris [url].
  // if modality is pinned to image, a video result must NOT be reused. no mutation of the input.
  throw new Error("not implemented");
}

export function deterministicPacket(request: Request): Packet {
  // a clip forces video. otherwise modality decides; "auto" with no clip means image.
  // image: imageUris.length > 0 -> image_edit with aspect match_input_image, else text_to_image with 1:1.
  // video: clip -> video_edit, 2+ images -> reference_to_video, 1 image -> image_to_video, none -> text_to_video.
  throw new Error("not implemented");
}

export function routeImage(operation: "text_to_image" | "image_edit", plannedFamily: string | null, endpoints: Record<string, { edit: string; textToImage: string }>, fallbackFamily: string): { endpoint: string; family: string; reason: string } {
  // use the planned family only if it exists in endpoints. otherwise use fallbackFamily and say so in reason.
  // image_edit picks the edit endpoint, text_to_image picks the textToImage endpoint.
  throw new Error("not implemented");
}
