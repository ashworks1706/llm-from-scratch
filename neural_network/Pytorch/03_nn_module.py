# building custom nn.Module classes
# understanding the base class for all neural network modules

# topics to cover:
# - what is nn.Module
# - __init__ and forward methods
# - registering parameters
# - registering buffers
# - state_dict and load_state_dict
# - train() vs eval() modes
# - module composition (modules containing modules)


# nn.module is the base class fsor all nns in pytorch 

# common terms in AI/ML:
# - model: a neural network architecture with parameters
# - layer: a building block of a neural network (e.g., linear layer, convolutional layer)
# - neuron: a single computational unit within a layer
# - parameter: a learnable weight or bias in the model
# - forward pass: the process of passing input data through the model to get output
# - backward pass: the process of computing gradients for parameters based on loss
# - activation function: a non-linear function applied to the output of a neuron or layer
# - loss function: a function that measures the difference between predicted and true values
# - optimizer: an algorithm that updates model parameters based on gradients
# - epoch: one complete pass through the training dataset   
# - batch size: the number of samples processed before the model is updated
# - overfitting: when a model learns the training data too well and performs poorly on new data
# - underfitting: when a model is too simple to capture the underlying patterns in the data
# - regularization: techniques used to prevent overfitting (e.g., dropout, weight decay)
# - dropout: a regularization technique where random neurons are ignored during training
# - weight decay: a regularization technique that adds a penalty to the loss function based on the size of the weights
# - batch normalization: a technique to normalize the inputs of each layer to improve training speed and stabiltiy
# - convolutional layer: a layer that applies convolution operations, commonly used in image processing
# - recurrent layer: a layer that processes sequential data, commonly used in natural language processing
# - transformer: a neural network architecture that uses self-attention mechanisms, commonly used in NLP tasks
# - self-attention: a mechanism that allows the model to weigh the importance of different parts of the input data
# - embedding: a dense vector representation of discrete variables, commonly used for words in NLP
# - fine-tuning: the process of taking a pre-trained model and adapting it to a specific task by training it on new data
# - transfer learning: the process of leveraging knowledge from one task to improve performance on a different but related task
# - hyperparameters: settings that control the training process (e.g., learning rate, batch size, number of epochs)
# - learning rate: a hyperparameter that controls the step size during optimization
# - gradient descent: an optimization algorithm that updates parameters to minimize the loss function
# - stochastic gradient descent (SGD): a variant of gradient descent that updates parameters using a subset of the data (mini-batch)
# - Adam: an optimization algorithm that combines the benefits of RMSProp and momentum
# - RMSProp: an optimization algorithm that adapts the learning rate for each parameter based on recent gradients
# - momentum: a technique that accelerates gradient descent by considering past gradients
# - early stopping: a technique to prevent overfitting by stopping training when performance on a validation set starts to gradients
# - validation set: a subset of the data used to tune hyperparameters and evaluate model performance during training
# - test set: a subset of the data used to evaluate the final model performance after training
# - cross-validation: a technique to assess model performance by splitting the data into multiple training and validation settings
# - data augmentation: techniques used to increase the diversity of the training data by applying transformations (e.g., rotation, flipping)
# - feature extraction: the process of transforming raw data into a set of relevant features for model training
# - dimensionality reduction: techniques used to reduce the number of features while preserving important information (e.g., PCA, t-SNE)
# - ensemble learning: combining multiple models to improve overall performance (e.g., bagging, boosting)
# - bagging: an ensemble technique that trains multiple models on different subsets of the data and averages their predictions
# - boosting: an ensemble technique that sequentially trains models to correct the errors of previous models
# - ROC curve: a graphical representation of a model's performance across different classification thresholds
# - AUC (Area Under the Curve): a metric that quantifies the overall performance of a classification model based on the ROC curve
# - precision: the ratio of true positive predictions to the total predicted positive
# - recall: the ratio of true positive predictions to the total actual positive
# - F1 score: the harmonic mean of precision and recall, providing a single metric for model performance
# - confusion matrix: a table that summarizes the performance of a classification model by showing true vs predicted labels


import torch 
import torch.nn as nn 

class SimpleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__() # always call parent __init__

        # register pparameters 
        self.weight = nn.Parameter(torch.randn(input_size))
        # the weights are the lines from input 
        self.bias = nn.Parameter(torch.randn(1))
        # the bias is added to the neuron

    def forward(self, x):
        # x : (batchsize, inputsize)
        # compute: output = sum(x * weight) + bias since
        # A single neuron computes:
        # output = w1*x1 + w2*x2 + w3*x3 + bias
        output = torch.sum(x * self.weight) + self.bias 
        # Matrix multiply (@) - DOESN'T WORK!
        # result = x @ weight  # ERROR or gives scalar dot product!
        # For @ to work: (a, b) @ (b, c) = (a, c)
        # But we have (3,) @ (3,) = scalar (dot product)
        # Result would be: 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 1.7 (single number)
        # use * when shapes match and you wat element by element 
        # use @ when we want matrix multiply in neural netwroks 
        # thatas why we can also use 
        # output = x @ self.weight + self.bias  since this is a dot product
        # dot product multiplies and sums in one operation 
        return output 

neuron = SimpleNeuron(input_size=3)
x = torch.tensor([1,2,3])

output = neuron(x) 

print(f"Input: {x}")
print(f"Weight: {neuron.weight}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output}")

print(f"\nTrainable parameters:")
for name, param in neuron.named_parameters():
    print(f"{name}: {param.shape}")



class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.w1 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b1 = nn.Parameter(torch.randn(hidden_size))
        self.w2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b2 = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        # x is batch size, inputsize
        # hidden layer 
        # since shapes won't match we do matrix mult in x and w1
        h = x @ self.w1 + self.b1 # linear transformation 
        h = torch.relu(h) # activation (non linearity)

        output = h @ self.w2 + self.b2 # linear transformation

        return output

model = TwoLayerNet(input_size=4, hidden_size=8, output_size=2)
# Input (4)
#        ↓
#     [Layer 1: w1, b1 + ReLU]
#        ↓
#     Hidden (8)
#        ↓
#     [Layer 2: w2, b2]
#        ↓
#     Output (2)
# Test it
x = torch.randn(3, 4)  # Batch of 3 samples, 4 features each
output = model(x)
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Can we train it?
print(f"All parameters require grad: {all(p.requires_grad for p in model.parameters())}")



