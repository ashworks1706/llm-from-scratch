# reverse-mode autograd: build a graph in the forward pass, then walk it backward applying the chain rule.
# each node stores its value, grad, parents, and a closure that pushes grad to the parents.
# order matters: process a node only after every consumer has added into its grad (reverse topo order).
# grads ACCUMULATE with +=, because one value can feed many nodes (x*x uses x twice).
# deps: stdlib only. compare against torch.autograd in the main block. see pytorch/02_autograd.py.
# explain: what breaks if backward uses = instead of += on parent grads?
# explain: why does a recursive backward blow the stack on long graphs, and what replaces it?

import math


class Value:
    def __init__(self, data: float, parents: tuple = ()):
        # store data, grad=0.0, parents, and a no-op _backward closure.
        pass

    def __add__(self, other):
        # return a new Value; its _backward adds out.grad into both parents.
        pass

    def __mul__(self, other):
        # local grads: d(a*b)/da = b, d(a*b)/db = a. accept python floats as other.
        pass

    def tanh(self):
        # local grad is 1 - tanh(x)^2, computed from the OUTPUT value.
        pass

    def backward(self):
        # build a topological order iteratively (no recursion), seed self.grad = 1.0,
        # then call each node's _backward in reverse order. O(nodes + edges).
        pass


if __name__ == '__main__':
    # build y = tanh(a*b + a) with a node used twice, call backward.
    # assert a.grad and b.grad match torch autograd to 1e-6.
    pass
