training covers how models get their weights, from pretraining to post training, and the systems and evals around it.

pretraining trains models on massive text corpora to predict next tokens. this is where models learn language patterns, facts and reasoning.

sft does supervised finetuning on instruction response pairs to teach models to follow instructions and have conversations. only computes loss on responses not instructions.

lora is parameter efficient finetuning using low rank matrices. trains less than 1 percent of parameters by injecting small adapters into attention layers.

rl implements dpo, ppo and grpo, plus rlvr rewards and the 2026 fixes around them. deepseekr1 builds on it for reasoning models.

distillation compresses a large teacher model into a smaller student, offline from soft labels and on-policy from the student own samples.

evals, speedrun, scalinglaws, datacuration, distributed and longcontext are the training systems side, measuring, tuning, scaling and feeding the training runs.
