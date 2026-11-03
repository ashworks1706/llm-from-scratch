this folder is for getting my programming grip back. every file is an unsolved practice file: concept comments on top, stubs with objectives below. solve it by hand, no ai, then answer the explain lines out loud.

for every topic: explain it from memory, implement the small example, connect it to one of my codebases. files that say "from sparky:", "from platform:" etc point at the real code i shipped. read that code after solving the toy, not before.

files that say "solve X first" are pointers into the rest of this repo (inference/, llama3/, rl/, pytorch/ ...). do the scaffold there, then come back for the reasoning stubs.

## by syllabus section

algorithms and coding interviews: dsa/ 00-24, plus python/05-08 for the basics it builds on.

languages and frameworks:
- rust/01-16: ownership, lifetimes, smart pointers, Send/Sync, trait objects, error enums, iterators and closures, async and cancellation, futures by hand, serde
- python/01-18: basics through generators, decorators, dataclasses, typing, asyncio, profiling, packaging, descriptors, aliasing
- typescript/01-14: basics through advanced types, unions, branded types, schema validation, event loop, Result vs throw, reducers, dependency injection

systems and backend: systems/01-32. os (01-09), networking (10-14), architecture (15-17), databases (18-23), search (24-26), distributed (27-32).

math and ml foundations, the programming side only: ml/01-13.

inference systems: inference/01-08, cuda/01-14, then the gpu and systems tracks in ../inference.

llm architecture and behavior: llm/01-11.

vision: vision/01-04.

reinforcement learning: rl/01-06.

## by project

sparky: rust/20-39 (request path, agent loop, tools, policy and confirmation, prompt budgets, config, retries, egress, trace events, sse, dependency direction), python/50-54 (evals and the baseline gate).

platform and godfather: python/20-39 (tenant scoping, scopes, module gates, service errors, jobs, limits, cache, ledger, checkout race, pod authorization, ssh certificates, cli login, scheduler, mcp tool pipeline, alembic dag, deploy and rollback).

product studio (halo): typescript/20-39 (intent routing, planner handles, context budget, moderation, tagged errors, service layers, provider jobs, exactly-once charges, webhooks, outbox, experiments, template exploration and cold start, taste ranking, eval scoring).

rise lab (risky_navigation): python/40-49 (delay buffer, window validity, fancy indexing, episode folds, normalization, baselines, sac pieces, scope bugs, sharded search).

piramid: rust/40-57 (distance slab, strategy registry, top-k heap, filters, resident slab, wal, compaction, manifest, quantization, device budget, kv blocks, scheduler, engine thread, sampling, retrieval hook, error mapping, config, benchmarking), cuda/20-24 (distance, top-k, int8 dot, device budget).

sportsitters and decision theater have no code here yet. systems/ covers most of their syllabus topics (race conditions, caching, query plans, indexes).

## running

python: python3 file.py. rust: rustc --edition 2021 file.rs, or drop it into a scratch cargo project when it says it needs tokio or serde. typescript: npx tsc --strict --noEmit file.ts, then run with tsx or node. cuda: nvcc file.cu.
