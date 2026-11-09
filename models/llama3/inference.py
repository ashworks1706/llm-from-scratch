# this file is distributed in phases for inference
# 1 -> Prefill phase : we process the entire prompt at once (parallel).
# 2 -> Decode Phase: We generate one token at a time (serial) using the KV Cache.
# 3 -> Temperature Sampling: We don't just pick the "best" word (which is boring); we pick "likely" words to add creativity.

import torch
import torch.nn.functional as F
import tiktoken
from .model import Llama
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
    # To calculate a cumulative total (e.g., "Top 90%"), we must line up the probabilities from largest to smallest.
    probs_sum = torch.cumsum(probs_sort, dim=-1) # We calculate the running total to find where we cross the threshold $p$.

    # creating mask for tokens to keep
    mask = probs_sum - probs_sort > p # We want to include the token that pushes us over the line. If we just did probs_sum > p, we 
    # would cut off Index 2 immediately because 0.95 > 0.9. But we need Index 2 to reach our 90% quota!
    probs_sort[mask] = 0.0

    # renormalizing the remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # PyTorch will crash because probabilities must sum to 1.0.

    # sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1) # sampling from the probabilities

    # retreive the original token index
    
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token

# If the model is unsure (flat distribution), the top 90% might include 1000 words. (High variety).
# If the model is sure (spiky distribution), the top 90% might include only 1 word. (High precision). 
# This prevents the model from saying gibberish just because it was "unlucky" with the random number generator.

def generate(model, tokenizer, prompt: str, max_new_tokens : int = 50, temperature: float = 0.7, top_p: float = 0.9, device: str = 'cuda'):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]

    # we need one kvcache object per layer, we also need to determien the maximum buffer size we might need
    max_len = seq_len + max_new_tokens + 10 # +10 safety buffer

    # grab dimension from the model config

    head_dim = model.layers[0].attention.head_dim
    n_kv_heads = model.layers[0].attention.n_kv_heads

    kv_cache_list = [
        KVCache(batch_size, max_len, n_kv_heads, head_dim, device)
        for _ in range(len(model.layers))
    ]
    
    # we process the entire prompt in parallel. the cache gets filled wiht kv of the prompt

    print(f"\n Prompt: {prompt}")
    print("Generaitng: ", end="", flush=True)

    with torch.no_grad():
        logits = model(tokens, start_pos=0, kv_cache_list=kv_cache_list) 

    # but what are logits? logits are essentially the raw, unnormalized 
    # predictions that a model generates before applying any activation function like softmax.

    # select the next token from the last position
    last_logit = logits[:,-1,:]

    if temperature > 0:
        probs = F.softmax(last_logit/temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        # greedy generation (pick the single best word)
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    # we loop one token at a time. we only pass the next_token to the model, the history we retrieved from the kv cache.

    generated_ids = [next_token.item()]
    curr_pos = seq_len # we are now at the position after the prompt

    for _ in range(max_new_tokens):
        # we pass only the last new token
        
        with torch.no_grad():
            logits = model(next_token, start_pos=curr_pos, kv_cache_list=kv_cache_list)

        last_logit = logits[:,-1,:] # get the logits for the last token

        if temperature > 0:
            probs = F.softmax(last_logit/temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            # greedy generation (pick the single best word)
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        decoded_word = tokenizer.decode(next_token.squeeze().tolist())
        print(decoded_word, end="", flush=True)

        generated_ids.append(next_token.item())
        curr_pos += 1
        print()
    return generated_ids




















