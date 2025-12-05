import torch
import torch.nn.functional as F
import tiktoken
from mixtral.model import Mixtral 
from mixtral.kv_cache import KVCache
from utils.config import Config

# Reuse the sampling function from before
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p 
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, device='cuda'):
    # 1. Setup
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # 2. Init Cache
    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]
    max_len = seq_len + max_new_tokens + 10
    
    # Grab dimensions from the model
    # Note: We access model.layers[0].attention...
    head_dim = model.layers[0].attention.head_dim
    n_kv_heads = model.layers[0].attention.n_kv_heads
    # initializing the KV cache as required 
    kv_cache_list = [
        KVCache(batch_size, max_len, n_kv_heads, head_dim, device)
        for _ in range(len(model.layers))
    ]

    print(f"Prompt: {prompt}")
    print("Generating: ", end="", flush=True)

    # 3. Prefill
    with torch.no_grad():
        logits = model(tokens, start_pos=0, kv_cache_list=kv_cache_list)
    
    # what is logit? the raw, unnormalized numerical scores produced by the final layer of the 
    # neural network for every possible word or token in the model's vocabulary
    last_logit = logits[:, -1, :]
    if temperature > 0:
        probs = F.softmax(last_logit / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    # 4. Decode Loop
    generated_ids = [next_token.item()]
    curr_pos = seq_len

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(next_token, start_pos=curr_pos, kv_cache_list=kv_cache_list)
        
        last_logit = logits[:, -1, :]
        
        if temperature > 0:
            probs = F.softmax(last_logit / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        decoded_word = tokenizer.decode([next_token.item()])
        print(decoded_word, end="", flush=True)
        
        generated_ids.append(next_token.item())
        curr_pos += 1

    return tokenizer.decode(generated_ids)
