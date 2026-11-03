# three signals: logs (events), metrics (aggregates), traces (one request across hops). structured logs
# are key=value or JSON so they can be filtered. a trace is a tree of spans: trace_id shared by all, each
# span has its own span_id and a parent_id; child duration is inside the parent's. RED metrics per
# service: Rate (req/s), Errors (failed fraction), Duration (latency distribution). report percentiles, not
# averages: the mean hides the tail, and percentiles cannot be averaged across hosts, so merge histograms.
# also solve mlops/06_observability_and_tracing.py and inference/systems/25_metrics_benchmarks_and_reliability.rs.
# from sparky: apps/engine/src/core/telemetry.rs builds a tracer provider that exports spans to Phoenix, and
# runtime/harness/agent/call/spans.rs records gen_ai.* and llm.token_count.* fields on model spans.
# from platform: core/log.py lifts key=value pairs of the request_log and access lines into JSON fields.
# explain: why does p99 of a fan-out of 10 calls depend mostly on the p99.9 of each call?

import contextvars
import time

current_span: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)


class Tracer:
    def __init__(self, clock=time.monotonic):
        # finished spans are appended to self.spans as dicts.
        pass

    def span(self, name: str):
        # context manager. new trace_id if no current span, else inherit it and set parent_id.
        # record start/end/duration, status "error" if the body raised (re-raise it), and always
        # restore the previous current span via the contextvar token.
        pass


def log_line(level: str, msg: str, **fields) -> str:
    # one JSON object with level, msg and the fields, plus trace_id/span_id when a span is current.
    # keys sorted for stable output. never raise on non-serializable values (use repr).
    pass


class Histogram:
    def __init__(self, bounds: list[float]):
        # upper bounds ascending plus an implicit +inf bucket; counts per bucket; total count.
        pass

    def observe(self, value: float) -> None:
        pass

    def percentile(self, p: float) -> float:
        # estimate p in (0, 100] from bucket counts; return the upper bound of the bucket holding the rank.
        pass

    def merge(self, other: "Histogram") -> "Histogram":
        # same bounds required (ValueError otherwise). this is how you aggregate across hosts.
        pass


def red(requests: list[tuple[float, bool]], window_s: float) -> dict:
    # requests = [(duration_s, ok)]. return {"rate": per second, "error_ratio": ..., "p50": ..., "p99": ...}.
    # empty list -> zeros, no division error.
    pass
