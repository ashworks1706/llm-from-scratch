trl (transformer reinforcement learning) library from huggingface for aligning language models with human preferences. while this codebase implements rl from scratch trl provides production ready trainers.

trl simplifies the complex process of rlhf (reinforcement learning from human feedback). covers supervised fine tuning, reward modeling, and preference optimization in standardized trainers.

sft trainer wraps supervised fine tuning with best practices. covers instruction dataset formatting, loss masking on prompts, and efficient training compared to manual loops.

dpo trainer implements direct preference optimization. covers training on chosen vs rejected pairs, reference model management, and beta parameter tuning without needing separate reward model.

ppo trainer implements proximal policy optimization. covers the full rlhf pipeline with reward model, on policy generation, and kl penalty to prevent model collapse.

reward trainer trains reward models from preference data. covers pairwise comparison datasets, bradley terry model, and outputting scalar rewards.

understanding trl after implementing from scratch helps you use these tools effectively and debug when things go wrong. you know what abstraction hides.
