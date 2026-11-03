// a module's public surface should be small and behavioral; callers get operations, not its tables, clients or clocks.
// dependency injection means a function receives what it needs (clock, store, http) instead of importing a concrete one.
// build the real wiring once at a composition root; everything below it takes dependencies as parameters.
// the payoff is testing with a fake clock and an in-memory store, no module mocks and no sleeping.
// ambient time and randomness are dependencies too: Date.now() inside a service makes it untestable.
// from halo: packages/api/src/identity/IdTokenVerifier.ts is an Effect service with layerWithVerifier(verifyToken) so tests inject a fake verifier.
// from halo: packages/api/src/http/outbound.ts withOutboundTimeout takes the timeout name and an onTimeout error factory as arguments.
// from halo: .agents/skills/coding-standards/references/modules-services-and-adapters.md defines composition root and adapters.
// this file does it with plain functions and interfaces, no Effect.
// explain: what does the deletion test say about a one-method interface that has exactly one implementation and no fake?
// explain: why is importing a singleton db client inside a service a hidden dependency, even if it is typed?

export interface Clock {
  now(): number;
}

export interface Store<V> {
  get(key: string): V | undefined;
  set(key: string, value: V): void;
  delete(key: string): void;
}

export interface TokenService {
  issue(userId: string): string;
  verify(token: string): { userId: string } | null;
}

export function makeFakeClock(start: number): Clock & { advance(ms: number): void } {
  // now() returns the current fake time; advance moves it forward. no real timers.
  throw new Error("not implemented");
}

export function makeMemoryStore<V>(): Store<V> {
  // a Map-backed store. get of a missing key is undefined; delete of a missing key is a no-op.
  throw new Error("not implemented");
}

export function makeTokenService(deps: { clock: Clock; store: Store<{ userId: string; expiresAt: number }>; ttlMs: number }): TokenService {
  // issue stores an opaque, unique token with expiresAt = now + ttlMs. verify returns null for unknown or expired tokens
  // and deletes an expired entry. only the returned interface is public: the store shape is not exposed.
  // uniqueness without Math.random: use a counter in the closure.
  throw new Error("not implemented");
}

export function buildApp(env: { ttlMs: number }): { tokens: TokenService } {
  // the composition root: the only place that picks the real clock (Date.now) and a concrete store.
  throw new Error("not implemented");
}
