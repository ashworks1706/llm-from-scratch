# integrating wandb with huggingface trainer
# automatic logging without manual wandb.log calls

# topics to cover:
# - WandbCallback with Trainer
# - report_to="wandb" in TrainingArguments
# - automatic metric logging
# - logging generation samples during training
# - gradient and parameter histograms
# - model checkpointing to wandb artifacts
# - resuming training from wandb artifacts
# - disabling wandb for debugging

# OBJECTIVE: add wandb to existing trainer based training
# understand automatic logging vs manual control
