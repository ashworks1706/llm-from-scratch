pytorch fundamentals implemented from scratch to understand what happens under the hood. each file builds core concepts needed for neural networks.

tensors are the fundamental data structure. covers creation, shapes, indexing, broadcasting and operations. understanding tensor manipulation is essential since everything in pytorch is tensors.

autograd implements automatic differentiation which is how pytorch computes gradients. covers computational graphs, requires_grad, backward pass and gradient accumulation. this is the magic that makes deep learning practical.

nn module is the base class for all models. implements custom modules, parameter registration, state management and the forward method pattern. understanding this clarifies what happens when you subclass nn module.

linear layer implements the basic y equals wx plus b transformation from scratch. covers weight initialization, matrix multiplication and how parameters connect to autograd.

activation functions add non linearity to networks. implements relu sigmoid tanh and others by hand. explains why linear transformations alone cannot solve complex problems.

loss functions measure prediction error. implements mse and cross entropy from scratch. covers why certain losses work for certain problems and numerical stability considerations.

optimizers update parameters using gradients. implements sgd and adam from scratch. shows how learning rate, momentum and adaptive methods work mathematically.

training loop ties everything together into a complete example. shows the full cycle of forward pass, loss computation, backward pass and parameter updates. includes validation and checkpointing.
