# training reward models for rlhf
# learning to score responses based on human preferences

# topics to cover:
# - RewardTrainer for preference datasets
# - dataset format (chosen, rejected pairs)
# - model outputs scalar reward not logits
# - bradley terry model for preferences
# - margin ranking loss
# - reward model evaluation
# - using trained reward model in ppo
# - reward hacking and how to prevent it

# OBJECTIVE: train reward model on preference data
# understand first stage of traditional rlhf pipeline
