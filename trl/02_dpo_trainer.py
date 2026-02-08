# direct preference optimization with DPOTrainer
# training models to prefer chosen over rejected responses

# topics to cover:
# - DPOTrainer initialization and config
# - dataset format (prompt, chosen, rejected columns)
# - reference model creation (frozen copy)
# - beta parameter (kl penalty strength)
# - comparing to manual dpo in rl/dpo_trainer.py
# - loss computation under the hood
# - label smoothing and loss_type variations
# - when dpo converges vs diverges

# OBJECTIVE: run dpo training using DPOTrainer abstraction
# understand what the trainer handles automatically
