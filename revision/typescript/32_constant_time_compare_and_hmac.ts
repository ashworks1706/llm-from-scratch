// comparing secrets with === leaks how many leading characters matched, through timing. compare in time that depends only on length.
// a webhook secret can be a shared header value (halo's revenuecat webhook) or an hmac of the body (signed urls, tokens).
// an hmac key should be purpose scoped: derive a subkey per context from one root secret by signing the context string.
// use the platform crypto (crypto.subtle in node 20+ and workers). never roll your own hash.
//
// from halo: packages/api/src/revenuecat/webhook-verification.ts (constantTimeEquals, revenueCatWebhookVerifier: 500 if the
// secret is not configured, 401 if missing or wrong) and infra/derived-hmac-key.ts (deriveHmacKey signs a context string with
// the root secret and imports the result as a new HMAC SHA-256 key). reproduce the compare and the derive/sign/verify trio.
//
// explain: why must the compare loop run over max(len) and not exit at the first difference?
// explain: why derive a key per purpose instead of using the root secret everywhere?

export function constantTimeEquals(left: string, right: string): boolean {
  // fold a diff: start with left.length XOR right.length, then OR in the XOR of each char code up to the LONGER length
  // (missing chars count as 0). equal only when diff is 0. no early return inside the loop.
  throw new Error("not implemented");
}

export async function deriveKey(rootSecret: string, context: string, usages: Array<"sign" | "verify">): Promise<CryptoKey> {
  // import rootSecret as an HMAC SHA-256 key, sign the utf8 context with it, import the raw signature bytes as the new key.
  // different contexts must give different keys; the same inputs must give the same key.
  throw new Error("not implemented");
}

export async function signHex(key: CryptoKey, message: string): Promise<string> {
  // HMAC SHA-256 of the utf8 message, lowercase hex.
  throw new Error("not implemented");
}

export async function verifyHex(key: CryptoKey, message: string, signatureHex: string): Promise<boolean> {
  // recompute the signature and compare with constantTimeEquals. a wrong length or non-hex input is false, never a throw.
  throw new Error("not implemented");
}
