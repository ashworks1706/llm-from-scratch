# we compute attention via sequences
# now if we have multiple requests coming in that have different lengths, 
# so to manage that we  add padding or empty token to batch so that it 
# matches the shape
# so obviosuly the padding tokens are not supposed to be computed by 
# attentions so thats why we had casual masking
# mask = [[1, 1, 0, 0, 0],  # first 2 are real
#         [1, 1, 1, 1, 1],  # all 5 are real
#         [1, 1, 1, 0, 0]]  # first 3 are real
# casual masking is basically a lower triangular matrix where token 0 only sees itself and then the second and so on


import torch
     
def pad_sequences(sequences, pad_value=0):
    
    # find max length
    max_len = max(len(seq) for seq in sequences) 
    
    # pad each sequence
    padded = []
    for seq in sequences:
        # how many to pad?
        num_padding = max_len - len(seq)
        
        # create padded sequence
        # append zeros to the sqeuences
        padded_seq = seq + [0] * num_padding
        
        padded.append(padded_seq)
    
    # create attention masks
    masks = []
    for seq in sequences:
        # 1s for real tokens, 0s for padding
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        masks.append(mask)
    
    # convert to tensors
    padded_tensor = torch.tensor(padded, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.long)
    
    return padded_tensor, mask_tensor

def batch_generate(model, prompts, tokenizer, max_new_tokens=50):
    # tokenize all prompts at once 
    tok_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    # add padding 
    pad_tensor, mask_tensor = pad_sequences(tok_prompts)
    # move to device 
    device = next(model.parameters()).device # why? because this gives all 
    # parameters, next() gets the first one and .device tells us where it is 
    pad_tensor = pad_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    # generate an autoregressive loop so for each new token, perform 
    for _ in range(max_new_tokens):
        # forward pass on all sequences at once and then sample next token for each 
        # and append to sequence
        # loop through generates sequences and decode them back 
        # forward pass on current sequences
        logits = model(pad_tensor) # shape: batch, seq
        # we get logits for last position of each sequence since we want 
        # (batch,vocabsize) from (batch,seqlen,vocabsize)
        next_token_logits = logits[:,-1,:] # not [-1] since that gives last sequence only
        # i.e (seq, vocabsize) but we want (batch,vocabsize) i.e last positiion of all sequences
        # [:,-1,: ] : -> all batches, -1 -> last position, : = all vocab
        probs = torch.softmax(next_token_logits, dim=-1) # convert to prob
        next_tokens = torch.multinomial(probs, num_samples=1) # sample one token per seq 

        # append new tokens to sequences 
        # pad_tensor is (batch, seq_len), next_tokens is (batch, 1)
        # concatenate along sequence dimension
        pad_tensor = torch.cat([pad_tensor, next_tokens], dim=-1)
        
        # step 5: extend attention mask (new tokens are real, not padding!)
        new_mask = torch.ones((mask_tensor.shape[0], 1), device=device, dtype=torch.long)
        mask_tensor = torch.cat([mask_tensor, new_mask], dim=1)
    
    # decode all sequences back to text
    generated_texts = []
    for seq in pad_tensor:
        text = tokenizer.decode(seq.tolist())  # Convert tensor to list
        generated_texts.append(text)
    
    return generated_texts
