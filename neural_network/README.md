this folder is for building neural network fundamentals from scratch to solidify my understanding of deep learning and pytorch. after building complex llms i want to go back to basics and implement everything by hand.

the learning plan follows a bottom up approach starting with pytorch fundamentals then progressing through different neural network architectures. each implementation is done from scratch to understand what happens under the hood.

pytorch fundamentals covers tensors, autograd, building custom modules, activation functions, loss functions and optimizers. the goal is to understand what happens when you call model forward or loss backward instead of treating pytorch as a black box.

basic neural networks implements simple feedforward networks starting from a single neuron up to mnist classification. includes manual backpropagation implementation to see how gradients flow through the network.

cnns covers convolutional neural networks for computer vision. implements convolution and pooling operations from scratch then builds image classifiers. helps understand why cnns work for images through concepts like local patterns and translation invariance.

lstms and rnns implement recurrent networks for sequence modeling. builds vanilla rnn and lstm cells from scratch to understand gates and sequential processing. includes comparison to transformers to see why attention mechanisms replaced recurrent architectures in modern nlp.

vaes covers variational autoencoders for generative modeling. implements compression and reconstruction to understand latent spaces and how to generate new data rather than just classify existing data.

the progression builds from fundamentals to specialized architectures, with each section reinforcing concepts used in the llm work. the focus is on implementation and understanding rather than just using libraries.
