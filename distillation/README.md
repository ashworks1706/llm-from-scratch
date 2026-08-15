knowledge distillation trains a smaller student model to mimic a larger teacher model. instead of learning from hard labels which are binary right or wrong, the student learns from the teachers soft probability distributions which contain richer information about relationships between tokens.

the student config defines a smaller architecture with fewer layers, smaller embedding dimensions, and fewer attention heads. this gives a model that is 5 to 10 times smaller than the teacher, around 1 billion parameters compared to 7 billion. the smaller model is faster and uses less memory but aims to retain most of the teachers capabilities.

temperature scaling is crucial for distillation. dividing logits by a temperature greater than 1 softens the probability distribution so wrong answers get higher probabilities. this reveals which wrong answers the teacher considers almost right, teaching the student about relationships and nuances rather than just the final answer.

the training loss combines soft loss and hard loss. soft loss uses kl divergence to match the students distribution to the teachers softened distribution. hard loss is regular cross entropy with true labels as a safety net in case the teacher makes mistakes. typically we use 90 percent soft and 10 percent hard.

the t squared scaling factor normalizes gradients since softmax derivatives scale with one over temperature. without this the soft loss would be too small compared to hard loss. distillation can be combined with lora where only small adapter layers are trained on the student rather than all parameters.

after distillation the student model runs much faster with similar quality to the teacher. this is useful for deploying to devices with limited resources or serving many users where inference cost matters. the student achieves around 90 percent of teacher performance at a fraction of the computational cost.
