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
Option 1: One sample at a time (batch_size=1)
✓ Memory efficient
✗ Very slow, noisy gradients

Option 2: Entire dataset (batch_size=N)
✗ Out of memory
✗ Slow, stuck in local minima

Option 3: Mini-batches (batch_size=32-256) ✓
✓ Fits in memory
✓ Fast (parallel computation)
✓ Good gradient estimates
✓ Noise helps generalization





































































































































































































