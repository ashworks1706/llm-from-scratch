supervised finetuning is the second stage after pretraining where we teach the model to follow instructions and have conversations. instead of predicting next tokens on random text, the model learns from instruction response pairs.

the dataset file handles instruction following data. each example has an instruction or question and a desired response. the key difference from pretraining is that we only compute loss on the response tokens not the instruction. this is done by masking the instruction portion so the model only learns to generate good responses.

the training script is similar to pretraining but uses this masked loss. we typically use a smaller learning rate since we are finetuning not training from scratch. the model already knows language from pretraining so we just need to teach it the instruction following format.

sft usually takes much less time than pretraining, often just a few hours or days depending on dataset size. the model goes from being a text completer to being a helpful assistant that can answer questions and follow instructions.

after sft the model can have coherent conversations but might not be fully aligned with human preferences. that is why we do reinforcement learning from human feedback as the next step.
