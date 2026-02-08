# using the Trainer class for simplified training
# trainer handles the training loop, evaluation, and logging

# topics to cover:
# - TrainingArguments for configuring training
# - Trainer class initialization with model and datasets
# - trainer.train() and what happens during training
# - evaluation during training
# - logging to tensorboard/wandb
# - saving checkpoints automatically
# - resuming from checkpoints
# - learning rate scheduling
# - gradient accumulation and mixed precision

# OBJECTIVE: train a small model using Trainer and understand the abstraction
# compare to writing manual training loop (like in pretraining/)
