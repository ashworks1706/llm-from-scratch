# Given preference pair: (prompt, y_win, y_lose)

# Loss = -log(σ(β * (log_prob(y_win) - log_prob(y_lose) - reference_log_ratio)))
# Where:
# - log_prob(y) = how likely our policy generates y
# - reference_log_ratio = keeps us close to SFT model
# - β = temperature (controls strength)
# - σ = sigmoid function

# Intuition: Increase probability of winning response, decrease probability of losing response!

# KL divergence is crucial because it prevents the model from dfriting too far 
# its basically to avoid model from reward hacking for ex,
# Prompt: "Write a poem"
# Model learns: "Reward model gives high scores to long responses"
# Model exploits: Generates infinite gibberish to maximize reward!
# the model hacks the reward by finding shortcuts but they arent actually good
# so KL penalty prevents this by measuring how differnet policy is from refenrece (SFT Model)
# Loss = reward_loss + β * KL_divergence


# LoRA must be used with RL it's even better 

# SFT Training:
#   Input: instruction + response
#    Process: Model LEARNS to generate the response
#    Loss: CrossEntropy (how well does model predict next tokens?)
    
#    Example:
#    Input: "What is 2+2?" + "The answer is 4"
#    Model learns: Given instruction, predict these exact tokens

# DPO Training:
#    Input: instruction + chosen_response AND instruction + rejected_response
#    Process: Model learns to PREFER chosen over rejected
#    Loss: Comparison loss (make chosen more likely than rejected)
    
#    Example:
#    Prompt: "What is 2+2?"
#    Chosen: "The answer is 4"
#    Rejected: "I don't know"
    
#    Model already KNOWS how to generate responses (from SFT)
#    Now we teach it: "your chosen response should be MORE likely than rejected"

import torch
from torch.utils.data import Dataset
import json

class DPODataset(Dataset):
    # format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    # prompt: the instruction/question
    # chosen: the preferred response (human ranked higher)
    # rejected: the worse response (human ranked lower)
    
    def __init__(self, data_path, tokenizer, max_length=512):
        # TODO: load json data
        # TODO: store tokenizer and max_length
        pass
    
    def __len__(self):
        # TODO: return dataset size
        pass
    
    def __getitem__(self, idx):
        # TODO: get prompt, chosen, rejected
        # TODO: format as: prompt + chosen, prompt + rejected
        # TODO: tokenize both
        # TODO: return: prompt_ids, chosen_ids, rejected_ids
        pass
