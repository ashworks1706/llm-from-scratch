distributed training covers how models get trained across many gpus. the inference folder covers serving, this covers the training side.

ddp copies the model on every gpu and averages gradients. zero and fsdp shard the optimizer state, gradients and weights so each gpu only holds a part. tensor parallel splits single matmuls across gpus. activation checkpointing trades compute for memory and fp8 training runs matmuls in 8 bit. with one gpu, ranks are simulated on cpu with the gloo backend.
