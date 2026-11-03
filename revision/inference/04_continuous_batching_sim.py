# static batching waits for the slowest request in the batch; finished slots sit idle. continuous
# batching refills a freed slot at the next step, so utilization stays high and queueing time drops.
# a step is one model forward: every running sequence advances one token (decode) or a prefill chunk.
# model it as a discrete-event loop over steps with a max_batch and a token-budget per step.
# deps: stdlib only. pointers: solve inference/systems/21_batching_and_streaming.rs and
# inference/systems/22_prefill_decode_scheduler.rs first.
# from piramid: apps/engine/model/src/inference/batching/scheduler.rs (Scheduler::plan) packs one decode token
# per running sequence first, then prefill chunks, then admits new sequences while below max_batch_size.
# reproduce that order on a toy, not the scheduler.
# explain: why does continuous batching help throughput and p95 latency but not single-request latency?
# explain: how does a long prompt in a shared step hurt the decode latency of everyone else?

from dataclasses import dataclass


@dataclass
class Job:
    arrival: int        # step at which it arrives
    prompt: int         # tokens to prefill (1 step per prefill_chunk tokens)
    out: int            # tokens to decode, one per step


def simulate_static(jobs: list[Job], max_batch: int, prefill_chunk: int) -> dict:
    # take up to max_batch arrived jobs, run them until ALL finish, then take the next batch.
    # return {"steps": total steps, "finish": per-job finish step list, "util": fraction of batch slots used}.
    pass


def simulate_continuous(jobs: list[Job], max_batch: int, prefill_chunk: int) -> dict:
    # each step: advance every running job by one decode token or one prefill chunk, free finished slots,
    # then admit waiting arrived jobs into free slots. same return shape. never exceed max_batch.
    pass


if __name__ == '__main__':
    # make 20 jobs with mixed out lengths (5..200). assert continuous steps <= static steps,
    # every job finishes exactly once, and no step runs more than max_batch jobs.
    pass
