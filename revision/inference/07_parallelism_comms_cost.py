# data parallel: replicate the model, split the batch, all-reduce gradients (training) or just add replicas (serving).
# tensor parallel: split each matmul across gpus; needs an all-reduce TWICE per layer on the critical path.
# pipeline parallel: split layers into stages; moves activations once per stage boundary but leaves bubbles.
# ring all-reduce time ~ 2 * (n-1)/n * bytes / link_bandwidth (+ latency terms). over pcie this can erase
# the compute saved by splitting; over nvlink it usually does not. the answer depends on bytes per token.
# deps: stdlib only. pointers: solve inference/gpu/19_nccl_and_multi_gpu.rs first; see
# inference/systems/35_disaggregated_serving.rs and 36_moe_inference_serving.rs for where comms returns.
# explain: why is tensor parallel kept inside one node while pipeline parallel crosses nodes?
# explain: at batch 1 decode, why can adding gpus make tokens/s worse?

def allreduce_seconds(bytes_: float, n: int, link_bw: float, latency: float = 5e-6) -> float:
    # ring all-reduce. n == 1 costs 0.
    pass


def tp_decode_step_seconds(params: float, bytes_per_param: float, layers: int, hidden: int, batch: int,
                           act_bytes: int, n: int, mem_bw: float, link_bw: float) -> float:
    # per-gpu weight read time = params * bytes_per_param / n / mem_bw (decode is memory-bound),
    # plus 2 all-reduces per layer of batch * hidden * act_bytes bytes each.
    pass


def best_tp_degree(candidates: list[int], **kw) -> int:
    # candidate with the smallest tp_decode_step_seconds for the given kwargs.
    pass


if __name__ == '__main__':
    # 70b fp16, 80 layers, hidden 8192, batch 1: compare nvlink (~4.5e11 B/s) vs pcie (~2.5e10 B/s) for n in 1,2,4,8.
    # print where the per-step time stops improving and say why in a comment.
    pass
