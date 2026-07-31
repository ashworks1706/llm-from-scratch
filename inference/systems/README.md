the systems  are the core inference engineering track. candle runs the model, but an inference service still needs to manage many requests with different prompt lengths, output lengths, memory needs and client behavior.

these  own request state, kv cache policy, scheduling, streaming, observability and overload handling. this is where rust is especially useful because concurrency, lifetimes and explicit state machines are part of the actual problem.

the goal is not to copy a production runtime feature for feature. the goal is to build and explain a small correct system that demonstrates the same engineering decisions.

tech stack used in this folder:

- tokio — the async runtime; request handling, streaming and scheduling all run as tasks inside it
- axum — the http server framework for the openai-style api surface in file 13
- tower-http — request tracing and logging middleware layered onto axum
- tracing and tracing-subscriber — structured logs and spans for latency, queueing and lifecycle observability
- candle-core and candle-transformers — the model and cache types the scheduler and request state wrap around; these  manage them, not reimplement them
- anyhow and thiserror — anyhow for `?` in the binaries, thiserror for a typed request and scheduling error enum
- std::collections (VecDeque, HashMap) — queueing and cache bookkeeping structures behind the scheduling and kv cache policy 
