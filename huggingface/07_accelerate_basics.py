# distributed training and mixed precision with accelerate
# scaling training across multiple gpus without changing code

# topics to cover:
# - Accelerator class initialization
# - accelerator.prepare() for model, optimizer, dataloader
# - automatic device placement
# - gradient accumulation steps
# - mixed precision (fp16/bf16) training
# - distributed data parallel (DDP)
# - gradient checkpointing for memory efficiency
# - accelerator.backward() vs loss.backward()
# - saving and loading distributed checkpoints

# OBJECTIVE: take single gpu training code and make it multi gpu ready
# understand what accelerate does behind the scenes
