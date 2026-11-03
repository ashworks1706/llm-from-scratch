// a service is an interface behind a tag. a layer says how to build it. code asks for the tag, never for the class.
// tests swap the layer, production swaps in the real one, and nothing else in the call chain changes.
// a missing service is a loud named error at the call site, not an undefined three calls later.
// providing a layer returns a NEW container, so one request cannot leak services into another.
//
// from halo: packages/api/src/credits/Credits.ts (class Credits extends Context.Service, Layer.succeed(Credits, operations)),
// planner/PlannerPromptCache.ts (layerUnavailable: read returns none, write does nothing) and
// generation/GenerationBilling.ts (make yields Credits, RevenueCat and GenerationPersistence from context).
// write the minimal version with no effect dependency.
//
// explain: why does a no-op PlannerPromptCache layer make the planner testable offline?
// explain: what is the cost of a service locator versus passing arguments, and when is passing arguments better?

export type Tag<S> = { readonly key: string; readonly __service?: S };
export type Layer<S> = { readonly tag: Tag<S>; readonly build: (services: Services) => S };

export class MissingServiceError extends Error {
  readonly _tag = "MissingServiceError";
  constructor(readonly key: string) {
    super(`missing service ${key}`);
  }
}

export function tag<S>(key: string): Tag<S> {
  // a tag is just its key. two tags with the same key name the same service.
  throw new Error("not implemented");
}

export class Services {
  provide<S>(layer: Layer<S>): Services {
    // build the service once with the services known so far, return a new Services that also holds it.
    // do not mutate this. a later layer for the same key replaces the earlier one (that is how a test override works).
    throw new Error("not implemented");
  }

  get<S>(t: Tag<S>): S {
    // throw MissingServiceError when absent. the same instance comes back on every call.
    throw new Error("not implemented");
  }
}

// after implementing: build a Credits tag with deduct(userId, n), a real layer and a recording fake layer,
// then show the same function runs against both without edits.
