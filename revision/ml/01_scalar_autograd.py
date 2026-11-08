# Reverse-mode autograd records a graph while the forward pass runs and then walks it backward with
# the chain rule. Each node stores its value, its gradient, its parent nodes and a small closure
# that pushes the node's gradient into the parents. Gradients accumulate with +=, because one value
# can feed many nodes and x * x uses x twice. A node may be processed only after every node that
# consumes it has added into its gradient, which is reverse topological order. The same idea with
# tensors is in pytorch/02_autograd.py.

import math


# Problem: Scalar autograd value.
#
# Implement a Value class that wraps one float and supports multiplication and backward. A Value
# holds data (the float), grad (the derivative of the final output with respect to this value,
# starting at 0.0), the tuple of parent Values it was computed from, and a _backward closure that
# adds this node's contribution into its parents' grads. A leaf has no parents and a no-op closure.
# Combine the methods so that the graph for any expression built from multiplication is correct,
# including graphs where one Value is used several times.
class Value:
    # Problem: Value constructor.
    #
    # data is the float to store and parents is a tuple of Values this one was computed from, empty
    # by default. Store data, set grad to 0.0, store parents, and set _backward to a function that
    # does nothing.
    #
    # Example: Value(3.0) has data 3.0, grad 0.0, no parents.
    def __init__(self, data: float, parents: tuple = ()):
        pass

    # Problem: Multiplication node.
    #
    # Return a new Value whose data is self.data * other.data and whose parents are self and other.
    # other may be a Value or a plain python float or int; wrap a plain number in a Value first. The
    # new node's _backward must add other.data * out.grad into self.grad and self.data * out.grad
    # into other.grad, using += so that repeated use of one value accumulates. Do not run backward
    # of the parents from inside this closure; backward() does the ordering.
    #
    # Example: a = Value(2.0), b = Value(3.0), c = a * b gives c.data 6.0. After c.backward(),
    # a.grad is 3.0 and b.grad is 2.0.
    #
    # Example: x = Value(3.0), y = x * x gives y.data 9.0. After y.backward(), x.grad is 6.0
    # because x is used twice and each use adds 3.0.
    #
    # Example: a = 2.0, b = 3.0, c = (a * a) * b gives 12.0. After c.backward(), a.grad is 12.0
    # (2 * a * b) and b.grad is 4.0 (a * a).
    def __mul__(self, other):
        pass

    # Problem: Backward pass.
    #
    # Compute the gradient of self with respect to every Value that self depends on and leave it in
    # their grad fields. First build a topological order of all nodes reachable through parents,
    # using an explicit stack or loop and not recursion, so a graph that is thousands of nodes deep
    # cannot overflow the python stack. Then set self.grad to 1.0 and call each node's _backward in
    # reverse topological order, so that a node runs only after every consumer of it has added into
    # its grad. Visit each node once even if it is reachable by several paths.
    #
    # Example: for c = a * b with a = 2.0 and b = 3.0, c.backward() sets c.grad to 1.0, a.grad to
    # 3.0 and b.grad to 2.0.
    #
    # Aim for O(nodes + edges) time.
    def backward(self):
        pass


if __name__ == '__main__':
    # Build y = tanh(a * b + a) so that a feeds two nodes, then call y.backward(). This needs
    # addition and tanh on Value as well, so add those methods first, each with its own local
    # gradient (tanh has derivative 1 - tanh(x)^2). Build the same expression with torch tensors
    # that require grad and call backward on it.
    # Assert that a.grad and b.grad match the torch gradients to within 1e-6.
    # Example: a = 0.5 and b = 2.0 give y = tanh(1.5) = 0.905148, a.grad = 0.542120 and
    # b.grad = 0.090353.
    pass
