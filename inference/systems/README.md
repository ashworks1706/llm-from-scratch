the systems lessons are the core inference engineering track. candle runs the model, but an inference service still needs to manage many requests with different prompt lengths, output lengths, memory needs and client behavior.

these lessons own request state, kv cache policy, scheduling, streaming, observability and overload handling. this is where rust is especially useful because concurrency, lifetimes and explicit state machines are part of the actual problem.

the goal is not to copy a production runtime feature for feature. the goal is to build and explain a small correct system that demonstrates the same engineering decisions.
