# sgd subtracts lr * g, and momentum keeps a velocity v = mu * v + g and subtracts lr * v.
# adam keeps first and second moment estimates and corrects their bias toward zero at small steps.
# adamw decays the weights directly with p -= lr * wd * p instead of adding wd * p into the gradient.
# optimizer state is per parameter, and steps run under no_grad.
# pytorch/07_optimizer.py covers the torch.optim interface.

import torch


class SGDMomentum:
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        # keep a zero-initialized velocity buffer per parameter with the same shape.
        pass

    @torch.no_grad()
    def step(self):
        # update each parameter in place and skip parameters whose grad is None.
        pass


class AdamW:
    def __init__(self, params: list, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # keep per-parameter m and v and one shared step counter.
        pass

    @torch.no_grad()
    def step(self):
        # update m and v, apply bias correction and decoupled decay, all in place with no new graph.
        pass


if __name__ == '__main__':
    # run 20 steps on a tiny regression with the hand-written optimizer and torch.optim from the same init.
    # assert the parameters match with torch.allclose and atol=1e-6.
    pass
