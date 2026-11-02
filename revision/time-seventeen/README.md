# time seventeen — independent OA preparation

## known constraints
from the recruitment emails: 6 questions, 90-minute hard limit, deadline **November 9 2026, 23:59 UTC** (16:59 MST). the test prohibits AI, other people's help and external answer services; permitted materials are scratch paper, pen and calculator. use this material **before** opening the real assessment, not during it.

the exact question types are not known. this track is a practice curriculum, not claimed insider knowledge.

## 01 — market mechanics from first principles

a market matches bids (willingness to buy) and asks (willingness to sell). the best bid is the highest buy offer; the best ask is the lowest sell offer. a market buy order consumes the cheapest asks first, but can run out of quantity at a given price.

example: asks 10 at $0.60, 20 at $0.65, 50 at $0.70. buy 25 → cost 10×0.60 + 15×0.65 = $15.75.

a binary YES contract pays $1 if its event happens and $0 otherwise. its quoted price may be read as a rough implied probability, but bid/ask spread, fees, liquidity and risk affect this interpretation.

## 02 — quantitative reasoning

if your estimated probability is `p` and you pay `q` for a YES share, expected net profit (no fees) is `p - q`. actual profit is either `1-q` or `-q`. positive expected value is not guaranteed profit.

for exhaustive mutually exclusive outcomes, exactly one YES share pays $1. buying a complete set for less than $1 locks a gross profit under the assumptions; buying above $1 locks a loss. real-world settlement rules, access and fees matter.

## 03 — practice questions (independent)

1. calculate total notional and VWAP for trades with repeated market IDs.
2. implement market-order execution with quantity limits and insufficient-liquidity handling.
3. explain why a 90% true-positive rate doesn't imply 90% posterior probability when base rates are low.
4. for binary YES prices $0.62 and NO prices $0.33, reason about buying a complete pair before fees.
5. design a reliable pipeline that extracts contracts and evidence from messy documents, validates outputs, handles retries and records evaluation metrics.
6. debug a short aggregation function with missing keys, incorrect mutation, and floating-point precision issues.

set a 90-minute timer for a fresh six-question variant of these categories. do not read solution hints during your timed practice. after the timer, explain assumptions, failure cases, and complexity.

## additional concepts
order books, depth, spread, market orders vs. limit orders, liquidity, market makers and adverse selection, probabilities vs. prices, fees, tail events, contract ambiguity, idempotency and data validation.

## preparation sources
[Prediction Markets — Wolfers & Zitzewitz](https://www.aeaweb.org/articles?id=10.1257/0895330041371321) · [NeetCode](https://neetcode.io/) · [existing Python lessons](../python/README.md).
