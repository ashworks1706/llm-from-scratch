serving turns the model runtime into a reliable system that can accept concurrent requests and expose predictable latency.

the server owns request validation, streaming, cancellation, backpressure and observability while the scheduler owns execution order. keeping these boundaries separate prevents http concerns from leaking into kernels and cache management.

multi gpu execution is last because it adds communication and placement problems on top of every previous lesson.
