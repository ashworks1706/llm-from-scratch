# serving as a service is turning a model from a function you call in a script into an api
# other systems can hit over the network. that means request handling, input validation,
# batching, health checks and a versioned endpoint, not just model.generate. bentoml covers
# the packaging and request side. this lesson deliberately stops at the framework boundary:
# for the actual high-performance engine (paged attention, continuous batching) it hands off
# to the inference folder, which is where vllm and the custom kernels live.

# LEARNING OBJECTIVES:
# - Turn a model into an http service with a typed request/response instead of a bare function
# - Understand what a service adds over a script: validation, health, versioning, concurrency
# - Serve the bento packaged in lesson 03 and call it over http
# - Add basic request validation and see why untrusted input needs it
# - Know the boundary: bentoml handles serving glue, the inference folder handles the engine
# - Trace a served request with lesson 06's tooling so production calls are observable

import bentoml


def main():
    # step 1: define a bentoml service wrapping the model from lesson 03
    # step 2: give it a typed api endpoint (request schema in, response schema out)
    # step 3: serve it locally and call the endpoint with a real http request
    # step 4: add input validation and send a bad request to watch it get rejected cleanly
    # step 5: note where you'd swap the naive generate for a vllm backend (inference folder)
    # step 6: add tracing so each served request shows up in langfuse/phoenix from lesson 06
    pass


if __name__ == "__main__":
    main()
