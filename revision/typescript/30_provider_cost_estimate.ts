// provider cost is a pure function of (model, usage) over a pricing table. unknown model or bad usage returns undefined, never 0.
// three billing shapes: per image, per output second, per million tokens. surcharges are separate fields so they stay auditable.
// money is rounded once, to micro dollars, at the end. a cost you cannot compute is recorded as unknown, not as free.
// measure cost per generation, then compare it to the credits the user was charged to see the margin.
//
// from halo: packages/api/src/generation/provider-cost.ts (MODEL_PRICING, estimateProviderCost, estimateImageCost,
// estimateVideoCost, estimateTokenProviderCost, roundUsd). reproduce the three estimators on a small table.
//
// explain: why return undefined for an unpriced model instead of throwing or returning 0?
// explain: why record provider USD on the generation row at charge time and not recompute it later?

type Pricing =
  | { model: string; billing: "image"; perImageUsd: number; highThinkingPerRequestUsd?: number }
  | { model: string; billing: "second"; perSecondUsd: number; resolutionRates?: Record<string, number>; referenceImages?: { included: number; perExtraUsd: number } }
  | { model: string; billing: "token"; inputPerMillionTokensUsd: number; outputPerMillionTokensUsd: number };

const DEFAULT_RESOLUTION = "768P";

export function roundUsd(value: number): number {
  // round to 6 decimals (micro dollars).
  throw new Error("not implemented");
}

export function estimateMedia(table: Pricing[], input: { model: string; images?: number; thinkingLevel?: string; outputSeconds?: number; referenceImageCount?: number; resolution?: string }): { providerUsd: number; surchargeUsd: number } | undefined {
  // image: images defaults to 1 and must be a positive integer; surcharge only when thinkingLevel is "high".
  // second: outputSeconds must be finite and > 0; the default or 768P resolution uses perSecondUsd, any other resolution
  // must exist in resolutionRates or the result is undefined. surcharge = max(0, refs - included) * perExtraUsd.
  // unknown model or a token-billed model here -> undefined.
  throw new Error("not implemented");
}

export function estimateTokens(table: Pricing[], input: { model: string; inputTokens: number; outputTokens: number }): { inputUsd: number; outputUsd: number } | undefined {
  // strip a routing suffix from both names before matching: everything from the first ":" onward.
  // counts must be finite and >= 0. price per token = per-million / 1_000_000.
  throw new Error("not implemented");
}
