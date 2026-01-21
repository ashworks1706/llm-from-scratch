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
    # add padding 
    # move to device 
    # generate an autoregressive loop so for each new token, perform 
    # forward pass on all sequences at once and then sample next token for each 
    # and append to sequence
    # loop through generates sequences and decode them back 
    return  
