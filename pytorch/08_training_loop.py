# complete training loop example

# Load data, Define model, Compute loss, backpropagate, Update weight, Track progress, validate
# L D C B U T V 


# mainly 
# forward pass : feed data -> compute predictions -> compute loss 
# backward pass : compute gradients of loss wrt all parameters 
# optimizer : update parameters using gradients 
# repeat 

# epoch -> one complete pass through dataset 
#   inner loop -> batches 
#       process one mini batch data:input features, targets: ground truth labels 

# why batches?
# can't load entire dataset in memory, 
# Option 1: One sample at a time (batch_size=1)
# ✓ Memory efficient
# ✗ Very slow, noisy gradients

# Option 2: Entire dataset (batch_size=N)
# ✗ Out of memory
# ✗ Slow, stuck in local minima

# Option 3: Mini-batches (batch_size=32-256) ✓
# ✓ Fits in memory
# ✓ Fast (parallel computation)
# ✓ Good gradient estimates
# ✓ Noise helps generalization


# data loader class from torch we use it because it does automatic batching which is nice 
# we shuffle data, why shuffle?
# Without shuffle:
# Batch 1: All class 0 samples
# Batch 2: All class 1 samples
# → Model sees patterns in order, learns slowly

# With shuffle:
# Batch 1: Mixed classes
# Batch 2: Mixed classes
# → Model gets diverse examples, learns faster

# Training mode
# model.train()
# - Dropout active (random neuron dropping)
# - BatchNorm uses batch statistics
# - Gradients computed

# Evaluation mode
# model.eval()
# - Dropout off (use all neurons)
# - BatchNorm uses running statistics
# - No gradients (faster, less memory)


# gradient clipping is also needed to prevent exploding gradients, by scaling down large gradeints while skeeping direction 


# why do we do validaiton seperately?
# because training data: model learns from it, validation data: check if model generalizes well on unseen data 
# if training loss is less but validation loss is high: that means the model is overfitting and not learning,
# u can try regularization, more data or simpler model 
# if the loss is NaN, it could because of leanring rate is too high or numerical instability
# the soluton to this would be to lower learning rate, gradient clipping and data validation 

# if the loss is not decreasing, maybe its because LR is too low, wrong Loss function, or data not noramlized
# usualyl the soluton to this is increasing LR, looking for data shapes, and normalizing them 
#
# if the model is overfitting, that means training loss is less but validation loss is hihg 
# solutions could be adding dropout, weight decay (l2 regularization), getting more data, or early stopping (stop when validation loss stops improving)


































































































































































































































