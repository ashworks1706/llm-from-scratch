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
    # print(Value(3.0).data, Value(3.0).grad)       # 3.0 0.0
    # a, b = Value(2.0), Value(3.0)
    # c = a * b
    # c.backward()
    # print(c.data, c.grad, a.grad, b.grad)         # 6.0 1.0 3.0 2.0
    # x = Value(3.0)
    # y = x * x
    # y.backward()
    # print(y.data, x.grad)                         # 9.0 6.0
    # a, b = Value(2.0), Value(3.0)
    # c = (a * a) * b
    # c.backward()
    # print(c.data, a.grad, b.grad)                 # 12.0 12.0 4.0
    pass
