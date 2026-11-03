# ttft = first token time - request arrival. tpot (inter-token latency) = (finish - first token) / (output tokens - 1).
# end-to-end latency = finish - arrival. throughput = total output tokens / wall time of the whole run.
# averages hide the tail: report p50, p95, p99. percentiles need a stated method (nearest-rank vs interpolated).
# queueing makes ttft explode under load while tpot stays flat; tpot growing means batch pressure on decode.
# deps: stdlib only. pointer: solve inference/systems/25_metrics_benchmarks_and_reliability.rs first.
# explain: why is goodput (requests meeting an slo) a better headline than raw tokens per second?
# explain: if p99 ttft is high but p50 is fine, name three causes and one metric to tell them apart.

from dataclasses import dataclass


@dataclass
class Req:
    arrival: float
    first_token: float | None   # None if it never produced a token
    finish: float | None        # None if it was cancelled or failed
    out_tokens: int


def percentile(values: list[float], p: float) -> float:
    # nearest-rank percentile, p in (0, 100]. empty input raises ValueError. do not mutate values.
    pass


def summarize(log: list[Req]) -> dict:
    # return ttft p50/p95, tpot p50/p95, e2e p95, throughput (tokens/s over first arrival .. last finish),
    # and counts of completed vs dropped. exclude incomplete requests from latency stats but count them.
    # out_tokens == 1 has no tpot; skip it in the tpot list.
    pass


def goodput(log: list[Req], ttft_slo: float, tpot_slo: float) -> float:
    # completed requests per second that met BOTH slos, over the same wall window.
    pass


if __name__ == '__main__':
    # hand-build 5 requests with known times, assert each metric exactly.
    pass
