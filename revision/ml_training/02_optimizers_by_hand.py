# sgd: p -= lr*g. momentum: v = mu*v + g, p -= lr*v. adam: bias-corrected moments m, v.
# adamw decays weights directly (p -= lr*wd*p), not via the grad.
# state is per param; steps run under no_grad.

import torch


# 1. SGDMomentum
# each step, per param: v = mu*v + p.grad; p -= lr*v.
class SGDMomentum:
    # 1a. __init__
    # store params, lr, momentum. one zero velocity per param (same shape, dtype, device).
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        pass

    # 1b. step
    # update in place (never rebind p or v). skip params with grad None; their v is unchanged.
    # ex: p=1.0, lr=0.1, mu=0.9, grad 1.0 twice -> step 1: v=1.0, p=0.9; step 2: v=1.9, p=0.71
    @torch.no_grad()
    def step(self):
        pass


# 2. AdamW
# t starts at 1:
#   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g*g
#   m_hat = m / (1 - b1^t);  v_hat = v / (1 - b2^t)
#   p -= lr*wd*p;  p -= lr * m_hat / (sqrt(v_hat) + eps)
# decay uses p from before the adam update and never touches the grad.
class AdamW:
    # 2a. __init__
    # store params + hyperparams. zero m, v per param (same shape, dtype, device).
    # one step counter shared by all params, starting at 0.
    def __init__(self, params: list, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        pass

    # 2b. step
    # increment t once per call, then update m, v, p in place. skip params with grad None.
    # ex: p=1.0, lr=0.1, betas (0.9, 0.999), wd=0.01, grad 0.5, first step ->
    #     m=0.05, v=0.00025, m_hat=0.5, v_hat=0.25; decay p=0.999; then p -= ~0.1 -> p ~0.899
    @torch.no_grad()
    def step(self):
        pass


if __name__ == '__main__':
    # tiny regression; two copies of the same init params. 20 steps hand optimizer vs torch.optim
    # (SGD same momentum, then AdamW same settings), same data. assert allclose(atol=1e-6) for each.
    pass
