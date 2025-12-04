# this file is distributed in phases for inference
# 1 -> Prefill phase : we process the entire prompt at once (parallel).
# 2 -> Decode Phase: We generate one token at a time (serial) using the KV Cache.
# 3 -> Temperature Sampling: We don't just pick the "best" word (which is boring); we pick "likely" words to add creativity.

import torch
import torch.nn.functional as F
import tiktoken
from .model import LLama
from .kv_cache import KVCache
from utils.config import Config


def sample_top_p(probs,p):
    # hyperparameters :
    # temperature : this controls randomness. Directly proportional to creativity
    # top-p : this controls the percentage of next probable words, preventing model from choosing complete nonsense. 
    # Directly proportional to vocabulary.
    # This means that a reasonably low p, like 0.8, and high temp will produce quite interesting outputs, because 
    # the model will only choose from the most likely words, but won't go for the most most likely. It's perfect 
    # for "creative" models, e.g., for writing fiction.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending = True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # creating mask for tokens to keep
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    # renormalizing the remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)

    # retreive the original token index
    
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token

 def generate(model, tokenizer, prompt: str, max_new_tokens : int = 50, temperature: float = 0.7, top_p = float = 0.9, device: str = 'cuda'):





























