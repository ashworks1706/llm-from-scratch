# warmup ramps lr from ~0 so adam's early moment estimates are not trusted too much.
# cosine decay then lowers lr smoothly to a floor. the schedule is a pure function of step.
# a schedule bug is silent: off-by-one at the warmup boundary, decay past the last step, wrong floor.
# call scheduler.step() once per optimizer step, not per epoch, unless you chose epoch granularity.
# deps: torch for the LambdaLR check. see pytorch/08_training_loop.py (it uses ReduceLROnPlateau, not warmup).
# explain: why do big batch / big lr runs diverge without warmup?
# explain: what happens to resumed training if the scheduler state is not checkpointed?

import torch


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    # linear warmup 0 -> base_lr over warmup steps, then cosine decay to min_lr at total.
    # clamp: steps past total return min_lr. warmup=0 must not divide by zero.
    # lr_at(warmup) should equal base_lr exactly.
    pass


def as_lambda(base_lr: float, warmup: int, total: int, min_lr: float):
    # return a multiplier function f(step) usable with torch.optim.lr_scheduler.LambdaLR,
    # which multiplies the optimizer's base lr. keep it consistent with lr_at.
    pass


if __name__ == '__main__':
    # assert lr_at(0) is small, lr_at(warmup) == base_lr, lr_at(total) == min_lr, and monotone decay after warmup.
    # drive a real LambdaLR for total steps and compare its lr to lr_at at every step.
    pass
