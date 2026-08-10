# scaling is what happens when one gpu or one process isn't enough: too many requests, or a
# batch job too big to fit. ray gives you a way to spread python work across workers, and ray
# serve builds serving on top of it with replicas and autoscaling. the real lesson is judgment:
# distribution adds real complexity (networking, partial failure, coordination), so you only
# reach for it when a single machine genuinely can't keep up. on a laptop you mostly study the
# model and simulate scale rather than run a cluster.

# LEARNING OBJECTIVES:
# - Understand ray's actor/task model for spreading python work across workers
# - Distinguish scaling out serving (more replicas) from scaling a single big job (parallel batch)
# - Deploy a model behind ray serve with multiple replicas and a request router
# - Reason about autoscaling: what signal triggers it and what the latency/cost tradeoff is
# - Know when distribution is NOT worth it and a single well-batched gpu wins
# - Connect back to the inference folder: request concurrency vs true model parallelism

import ray
from ray import serve


def main():
    # step 1: start a local ray runtime and run a trivial task/actor across workers
    # step 2: wrap a small model in a ray serve deployment with 2 replicas
    # step 3: send concurrent requests and watch them spread across replicas
    # step 4: reason about an autoscaling policy: min/max replicas and the trigger metric
    # step 5: measure added latency from the routing layer vs a single in-process call
    # step 6: write down the threshold where you'd actually go distributed for this workload
    pass


if __name__ == "__main__":
    main()
