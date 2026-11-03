// the planner output is a proposal. a pin is a setting the user fixed explicitly, and a pin outranks the model.
// a hard conflict rejects the whole proposal and the rules plan runs instead (decision: constrained-fallback).
// a soft mismatch is repaired in place and recorded as an override (decision: accepted).
// rule: repair when the meaning survives, replace when it does not. a video plan on an image-bound turn cannot be repaired.
//
// from halo: packages/api/src/routing/constraints.ts (boundModality, packetConstraintViolation) and
// routing/turn-constraints.ts (enforceConstraints, constrainPlannedTurn). reproduce the two layers on toy types.
//
// explain: why is a wrong aspect ratio repairable but a wrong modality not?
// explain: what is the override log for, and who benefits when it is stored on the generation row?

type Pins = { modality: "auto" | "image" | "video"; aspectRatio?: string; seconds?: number };
type Packet =
  | { kind: "image"; operation: "text_to_image" | "image_edit"; imageUris: string[]; aspectRatio: string }
  | { kind: "video"; operation: "text_to_video" | "image_to_video" | "reference_to_video" | "video_edit"; imageUris: string[]; seconds: number };
type Override = { field: string; from: string; to: string };

const DEFAULT_SECONDS = 5;
const MIN_SECONDS = 5;

export function boundModality(pins: Pins, hasClip: boolean): "image" | "video" | undefined {
  // an attached clip binds the turn to video. otherwise a non-auto pin binds it. auto means unbound.
  throw new Error("not implemented");
}

export function violation(packet: Packet, pins: Pins, hasClip: boolean, hasImage: boolean): string | null {
  // return a reason string for a hard conflict, null when there is none.
  // conflicts: bound modality differs from packet.kind; image_edit with no imageUris and no attached image;
  // clip attached but the video operation reads no clip; video_edit with no clip anywhere.
  throw new Error("not implemented");
}

export function enforcePins(packet: Packet, pins: Pins): { packet: Packet; overrides: Override[] } {
  // image: a pinned aspectRatio replaces the planned one. video: unpinned seconds clamp into [MIN_SECONDS, DEFAULT_SECONDS].
  // push one Override per field that actually changed. return the same packet object when nothing changed.
  throw new Error("not implemented");
}
