an inference runtime coordinates model execution for many requests instead of running one prompt from beginning to end.

these lessons separate request state from model state, combine active sequences into continuous decode batches and schedule prefill work without blocking token generation for every other request.

cuda graphs and speculative decoding are added only after the basic scheduler is measurable and correct because both optimizations depend on stable execution and cache ownership.
