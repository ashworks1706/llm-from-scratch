# reverse-mode autograd: record the graph on forward, walk it backward with the chain rule.
# grads accumulate with += (x * x uses x twice). run nodes in reverse topological order.

import math


# 1. Value
# wraps one float; supports * and backward. correct even when one Value is used several times.
class Value:
    # 1a. __init__
    # store data, grad = 0.0, parents, and _backward = no-op. leaf has no parents.
    def __init__(self, data: float, parents: tuple = ()):
        pass

    # 1b. __mul__
    # return Value(self.data * other.data, parents (self, other)). wrap plain int/float in Value.
    # _backward: self.grad += other.data * out.grad; other.grad += self.data * out.grad.
    # don't call parents' backward from the closure; backward() handles ordering.
    def __mul__(self, other):
        pass

    # 1c. backward
    # build topo order of all reachable nodes iteratively (no recursion: deep graphs must not
    # overflow), visiting each node once. set self.grad = 1.0, call _backward in reverse topo order.
    # O(nodes + edges).
    def backward(self):
        pass


# ---- try it ----
if __name__ == '__main__':
    # assert abs(Value(3.0).data - 3.0) < 1e-6 and abs(Value(3.0).grad - 0.0) < 1e-6
    # a, b = Value(2.0), Value(3.0)
    # c = a * b
    # c.backward()
    # assert abs(c.data - 6.0) < 1e-6 and abs(c.grad - 1.0) < 1e-6 and abs(a.grad - 3.0) < 1e-6 and abs(b.grad - 2.0) < 1e-6
    # x = Value(3.0)
    # y = x * x
    # y.backward()
    # assert abs(y.data - 9.0) < 1e-6 and abs(x.grad - 6.0) < 1e-6
    # a, b = Value(2.0), Value(3.0)
    # c = (a * a) * b
    # c.backward()
    # assert abs(c.data - 12.0) < 1e-6 and abs(a.grad - 12.0) < 1e-6 and abs(b.grad - 4.0) < 1e-6
    pass
