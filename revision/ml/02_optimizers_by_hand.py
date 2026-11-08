# Plain SGD subtracts lr * g from each parameter. Momentum keeps a velocity v = mu * v + g and
# subtracts lr * v. Adam keeps a first moment estimate m and a second moment estimate v of the
# gradient and corrects their bias toward zero at small step counts. AdamW decays the weights
# directly with p -= lr * wd * p instead of adding wd * p into the gradient. Optimizer state is kept
# per parameter, and steps run under no_grad. pytorch/07_optimizer.py covers the torch.optim
# interface.

import torch


# Problem: SGD with momentum.
#
# Implement an optimizer over a list of torch tensors that have requires_grad set. params is the
# list of parameters, lr is the learning rate and momentum (mu) is the velocity decay. Each step
# computes v = mu * v + p.grad, then p -= lr * v, for every parameter.
class SGDMomentum:
    # Problem: SGDMomentum constructor.
    #
    # Store params, lr and momentum. Create one velocity buffer per parameter, filled with zeros,
    # with the same shape, dtype and device as the parameter.
    def __init__(self, params: list, lr: float, momentum: float = 0.9):
        pass

    # Problem: SGDMomentum step.
    #
    # Update every parameter in place from its current grad, under no_grad so no graph is built.
    # Skip any parameter whose grad is None and leave its velocity unchanged. Use in-place
    # operations on the parameter and the velocity, never rebind them.
    #
    # Example: one scalar parameter p = 1.0, lr = 0.1, momentum 0.9, and a grad of 1.0 on two
    # successive steps. Step 1 gives v = 1.0 and p = 0.9. Step 2 gives v = 0.9 * 1.0 + 1.0 = 1.9
    # and p = 0.9 - 0.1 * 1.9 = 0.71.
    @torch.no_grad()
    def step(self):
        pass


# Problem: AdamW.
#
# Implement the AdamW optimizer over a list of tensors. lr is the learning rate, betas is the pair
# (beta1, beta2) of decay rates for m and v, eps is added to the denominator for stability, and
# weight_decay is the decoupled decay coefficient. One step does, for step counter t starting at 1:
# m = beta1 * m + (1 - beta1) * g, v = beta2 * v + (1 - beta2) * g * g, m_hat = m / (1 - beta1^t),
# v_hat = v / (1 - beta2^t), p -= lr * wd * p, and p -= lr * m_hat / (sqrt(v_hat) + eps). The decay
# uses the parameter value from before the Adam update and never touches the gradient.
class AdamW:
    # Problem: AdamW constructor.
    #
    # Store params and the hyperparameters. Create zero-filled m and v buffers per parameter with
    # the same shape, dtype and device, and one step counter shared by all parameters, starting at 0.
    def __init__(self, params: list, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        pass

    # Problem: AdamW step.
    #
    # Increase the shared step counter once per call, then update m, v and each parameter in place
    # under no_grad, so no new graph is created. Skip parameters whose grad is None.
    #
    # Example: one scalar parameter p = 1.0 with lr = 0.1, betas (0.9, 0.999), weight_decay 0.01,
    # and a grad of 0.5 on the first step. m = 0.05, v = 0.00025, m_hat = 0.5, v_hat = 0.25. The
    # decay gives p = 1.0 - 0.1 * 0.01 * 1.0 = 0.999, then the Adam update subtracts
    # 0.1 * 0.5 / (0.5 + 1e-8), which is about 0.1, so p is about 0.899.
    @torch.no_grad()
    def step(self):
        pass


if __name__ == '__main__':
    # Create a tiny regression problem (for example a linear model fit to random data). Make two
    # copies of the same initial parameters. Run 20 steps with the hand-written optimizer on one copy
    # and 20 steps with torch.optim (SGD with the same momentum, then AdamW with the same settings)
    # on the other, using the same data each step.
    # Assert that the parameters of the two copies match with torch.allclose and atol=1e-6, for
    # each optimizer.
    pass
