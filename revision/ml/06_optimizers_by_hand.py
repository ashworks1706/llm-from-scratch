# sgd: p -= lr * g. momentum keeps a velocity: v = mu*v + g; p -= lr*v.
# adam keeps first and second moment estimates m, v, and corrects their bias toward zero at small t.
# adamw decays weights directly (p -= lr*wd*p), NOT by adding wd*p into the gradient.
# optimizer state is per-parameter and must survive checkpoints; steps happen under no_grad.
# deps: torch. see pytorch/07_optimizer.py. verify against torch.optim in the main block.
# explain: why does adam need bias correction at step 1, and what does the first update look like?
# explain: how much extra memory does adam cost per parameter, in fp32?

import torch


class SGDMomentum:
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        # keep a velocity buffer per parameter, same shape, zero init.
        pass

    @torch.no_grad()
    def step(self):
        # update every param that has a grad, in place. skip params with grad None.
        pass


class AdamW:
    def __init__(self, params: list, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # per-param m, v and a shared step counter t.
        pass

    @torch.no_grad()
    def step(self):
        # m, v update, bias-corrected, decoupled weight decay. all in place, no new graph.
        pass


if __name__ == '__main__':
    # run 20 steps on a tiny regression with your optimizer and torch.optim from the same init.
    # assert parameters match with torch.allclose(atol=1e-6).
    pass
