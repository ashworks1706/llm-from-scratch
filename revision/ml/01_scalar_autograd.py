# reverse-mode autograd builds a graph in the forward pass and walks it backward with the chain rule.
# each node stores its value, grad, parents and a closure that pushes grad to the parents.
# grads accumulate with +=, because one value can feed many nodes and x * x uses x twice.
# process a node only after every consumer has added into its grad, which is reverse topological order.
# pytorch/02_autograd.py shows the same idea with tensors.

import math


class Value:
    def __init__(self, data: float, parents: tuple = ()):
        # store data, grad of 0.0, parents and a no-op _backward closure.
        pass

    def __mul__(self, other):
        # local grads are d(a*b)/da = b and d(a*b)/db = a, and python floats are accepted as other.
        pass

    def backward(self):
        # build a topological order iteratively so a long graph cannot overflow the stack.
        # seed self.grad with 1.0 and call each _backward in reverse order, in O(nodes + edges).
        pass


if __name__ == '__main__':
    # build y = tanh(a * b + a) so that a feeds two nodes, then call backward.
    # assert a.grad and b.grad match torch autograd to 1e-6.
    pass
