# engineering fluency — revision

this is where i rebuild the ability to implement things independently across python, typescript, rust and cuda. the goal isn't knowing API names; it's understanding the data flow and being able to write, debug and measure working systems without an assistant producing the first solution.

## approach

every lesson follows the same pattern as the existing [from-scratch tracks](../README.md):

1. **why does this exist?** begin with a concrete problem rather than a vocabulary list.
2. **mental model.** describe what values and memory actually do.
3. **trace.** calculate or execute one small case by hand.
4. **implement.** write from a blank file before viewing a reference.
5. **break it.** run edge cases and diagnose failures.
6. **explain it.** compare design, complexity, ownership or measurements.

the source material here is written as a *learning guide*, not as solutions to copy. make your own implementations in a separate working directory or branch.

## roadmap

| track | current | what proficiency means |
|---|---:|---|
| [python](python/README.md) | 2/5 | data structures, idioms, algorithms, tests and async |
| [typescript](typescript/README.md) | 2/5 | types + JS runtime, collections, APIs and concurrency |
| [rust](rust/README.md) | 2/5 | ownership, traits, errors, collections and Tokio |
| [cuda](cuda/README.md) | 1/5 | correct kernels, data movement, reduction, GEMM, profiling |
| [time seventeen OA](time-seventeen/README.md) | immediate | independent quantitative + coding reasoning |
| [practice & review](practice/README.md) | ongoing | error journal, spaced rewrites and checkpoints |

until the Nov 9 2026 deadline, prioritize python and the time seventeen track. don't use AI or outside help during the actual assessment; its instructions explicitly prohibit that.

## definition of done

- [ ] describe the problem and invariants without notes.
- [ ] write the implementation without generated code.
- [ ] supply tests for empty inputs, boundaries and failure paths.
- [ ] explain complexity and why this approach fits.
- [ ] rewrite failures from memory the next day and one week later.

## cross-language project ladder

1. **python:** ingest and summarize trades, implement VWAP and order execution with tests.
2. **typescript:** build a typed, bounded-concurrency API aggregation service.
3. **rust:** implement an in-memory bounded cache and async request pipeline.
4. **cuda:** implement vector add, reduction and tiled matmul; compare with CPU and cuBLAS.

## source links

- [existing python reference implementations](../inference/reference/)
- [existing candle rust lessons](../inference/candle/)
- [existing gpu lessons](../inference/gpu/)
- [master syllabus](https://app.notion.com/p/344e4acf846e8057bc90d35e01d0bd67) — reference only; this revision lives on GitHub.
