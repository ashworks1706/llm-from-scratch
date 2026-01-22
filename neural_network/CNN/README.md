convolutional neural networks for computer vision implemented from scratch. cnns are specialized for processing grid like data such as images.

convolution implements the core sliding window operation with learnable kernels. covers how filters detect features like edges and patterns. explains weight sharing and why convolutions work better than fully connected layers for images.

pooling implements downsampling operations to reduce spatial dimensions. covers max pooling and average pooling. reduces computation and provides translation invariance.

simple cnn builds a complete network for mnist digit classification. stacks convolutional and pooling layers followed by fully connected layers. shows the typical architecture pattern for image classification.

resnet block implements residual connections which solved the vanishing gradient problem in deep networks. the skip connections allow gradients to flow directly through the network. this innovation enabled networks with hundreds of layers.

image classifier trains on cifar10 which has color images of real objects. includes data augmentation, batch normalization and dropout. shows how to build and train a practical image classification system.
