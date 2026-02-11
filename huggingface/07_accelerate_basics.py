# this is part of distrbuted training, how coudl u use more gpus instead of ur one single broke ass laptop gpu ?

# thats hwy we use Accelerate 

# we work with device management, distributed data parallele, mixed precision, gradient accumulation, gradient checkpointing 



# we DONT call .to(device ) in acclerated at all 

# gradinet checkpointing isj ust trade compute for memory, we dont store activatisn during forward pass 
# and we recompute during backward ppass, useful for very large models











