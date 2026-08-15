# Inference script for Kimi K2
# Demonstrates how to generate text using the model

import torch
import torch.nn.functional as F
from .model import Kimi 
from .kv_cache import KVCache


def sample_top_p(probs, p):
    """
    Nucleus (top-p) sampling - sample from smallest set of tokens with cumulative probability >= p
    
    This is better than top-k because it adapts to the probability distribution:
    - When model is confident (sharp distribution), samples from few tokens
    - When model is uncertain (flat distribution), samples from many tokens
    
    Args:
        probs: Probability distribution over vocabulary (Vocab_Size,)
        p: Cumulative probability threshold (e.g., 0.9)
    
    Returns:
        Sampled token ID
    
    Example:
    - Probs: [0.5, 0.3, 0.1, 0.05, 0.05]
    - p=0.9: Include tokens until sum >= 0.9
    - Nucleus: {token_0, token_1, token_2} with probs [0.5, 0.3, 0.1] (sum=0.9)
    - Set others to 0, renormalize, sample from [0.5/0.9, 0.3/0.9, 0.1/0.9, 0, 0]
    """
    
    # Step 1: Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Step 2: Compute cumulative sum
    # [0.5, 0.3, 0.1, 0.05, 0.05] -> [0.5, 0.8, 0.9, 0.95, 1.0]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Step 3: Create mask for tokens to remove
    # We remove tokens where (cumsum - current_prob) > p
    # This keeps the smallest set where cumsum <= p + current_token
    mask = probs_sum - probs_sort > p
    # Example: [False, False, False, True, True] for p=0.9
    
    # Step 4: Zero out masked probabilities
    probs_sort[mask] = 0.0
    
    # Step 5: Renormalize (sum to 1 again)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Step 6: Sample from filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Step 7: Map back to original token IDs
    # We sampled from sorted indices, need to get actual vocabulary IDs
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token


def generate(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_p=0.9, 
    device='cuda'
):
    """
    Generate text continuation for a given prompt.
    
    Two-phase generation:
    1. Prefill: Process entire prompt in parallel, fill KV cache
    2. Decode: Generate tokens one at a time, reusing cache
    
    Args:
        model: Kimi model instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling randomness (higher = more random)
                    0.0 = greedy (always pick most likely)
                    1.0 = sample from full distribution
                    >1.0 = more uniform (creative but incoherent)
        top_p: Nucleus sampling threshold (0.9 = use top 90% probability mass)
        device: 'cuda' or 'cpu'
    
    Returns:
        Generated text (str)
    """
    
    # ============================================
    # SETUP
    # ============================================
    
    # Encode prompt to token IDs
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    # Shape: (1, prompt_length)
    
    batch_size = tokens.shape[0]
    prompt_len = tokens.shape[1]
    
    # Calculate maximum sequence length we might reach
    max_len = prompt_len + max_new_tokens + 10  # +10 for safety
    
    # ============================================
    # INITIALIZE KV CACHE
    # ============================================
    # Create one cache per layer
    # Each cache stores K and V for all tokens we've processed
    
    head_dim = model.config.dim // model.config.num_heads
    n_kv_heads = model.config.num_kv_heads
    
    kv_cache_list = [
        KVCache(batch_size, max_len, n_kv_heads, head_dim, device)
        for _ in range(model.config.num_layers)
    ]
    
    print(f"Prompt: {prompt}")
    print("Generating: ", end="", flush=True)
    
    # ============================================
    # PHASE 1: PREFILL
    # ============================================
    # Process the entire prompt in one forward pass
    # This fills the KV cache with keys/values for all prompt tokens
    # Much faster than processing tokens one-by-one!
    
    with torch.no_grad():  # No gradient computation needed for generation
        logits = model(tokens, start_pos=0, kv_cache_list=kv_cache_list)
    
    # logits shape: (1, prompt_len, vocab_size)
    # We only care about the LAST position - that's where we predict the next token
    last_logit = logits[:, -1, :]  # (1, vocab_size)
    
    # Sample the first generated token
    if temperature > 0:
        # Apply temperature scaling (higher temp = more random)
        # Temperature divides logits, making distribution more/less peaked
        probs = F.softmax(last_logit / temperature, dim=-1)
        next_token = sample_top_p(probs[0], top_p)  # Remove batch dim
        next_token = next_token.unsqueeze(0)  # Add back for model input
    else:
        # Greedy decoding (always pick most likely)
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
    
    # ============================================
    # PHASE 2: AUTOREGRESSIVE GENERATION
    # ============================================
    # Generate tokens one at a time
    # Each iteration:
    # 1. Feed previous token to model
    # 2. Model uses cached K/V from all previous tokens (fast!)
    # 3. Compute attention only for new token
    # 4. Sample next token
    # 5. Repeat
    
    generated_ids = [next_token.item()]
    curr_pos = prompt_len  # Current position in sequence
    
    for step in range(max_new_tokens - 1):  # -1 because we already have first token
        with torch.no_grad():
            # Forward pass with single token
            # start_pos tells model where this token is in the sequence
            # Cache automatically grows with each step
            logits = model(next_token, start_pos=curr_pos, kv_cache_list=kv_cache_list)
        
        # Get logits for next token prediction
        last_logit = logits[:, -1, :]  # (1, vocab_size)
        
        # Sample next token
        if temperature > 0:
            probs = F.softmax(last_logit / temperature, dim=-1)
            next_token = sample_top_p(probs[0], top_p)
            next_token = next_token.unsqueeze(0)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)
        
        # Decode and print token immediately (streaming output)
        decoded_word = tokenizer.decode([next_token.item()])
        print(decoded_word, end="", flush=True)
        
        # Add to generated sequence
        generated_ids.append(next_token.item())
        curr_pos += 1
        
        # Optional: Stop if we generate end-of-sequence token
        # if next_token.item() == tokenizer.eos_token_id:
        #     break
    
    print()  # Newline after generation
    
    # Decode full generated sequence
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


