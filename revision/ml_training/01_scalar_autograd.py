# reverse-mode autograd: record the graph on forward, walk it backward with the chain rule.
# grads accumulate with += (x * x uses x twice). run nodes in reverse topological order.

import math


# 1. Value
# wraps one float; supports * and backward. correct even when one Value is used several times.
class Value:
    # 1a. __init__
    # store data, grad = 0.0, parents, and _backward = no-op. leaf has no parents.
    # ex: Value(3.0) -> data 3.0, grad 0.0, no parents
    def __init__(self, data: float, parents: tuple = ()):
        pass

    # 1b. __mul__
    # return Value(self.data * other.data, parents (self, other)). wrap plain int/float in Value.
    # _backward: self.grad += other.data * out.grad; other.grad += self.data * out.grad.
    # don't call parents' backward from the closure; backward() handles ordering.
    # ex: a=2, b=3, c=a*b -> c 6.0; after c.backward(): a.grad 3.0, b.grad 2.0
    # ex: x=3, y=x*x -> y 9.0, x.grad 6.0.  a=2, b=3, c=(a*a)*b -> 12.0, a.grad 12.0, b.grad 4.0
    def __mul__(self, other):
        pass

    # 1c. backward
    # build topo order of all reachable nodes iteratively (no recursion: deep graphs must not
    # overflow), visiting each node once. set self.grad = 1.0, call _backward in reverse topo order.
    # ex: c = a*b, a=2, b=3 -> c.grad 1.0, a.grad 3.0, b.grad 2.0
    # O(nodes + edges).
    def backward(self):
        pass


if __name__ == '__main__':
    # add + and tanh to Value (tanh' = 1 - tanh(x)^2). build y = tanh(a*b + a), y.backward(),
    # compare a.grad, b.grad to torch within 1e-6.
    # ex: a=0.5, b=2.0 -> y = tanh(1.5) = 0.905148, a.grad 0.542120, b.grad 0.090353
    pass
