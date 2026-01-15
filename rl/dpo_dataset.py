import torch
from torch.utils.data import Dataset
import json
import tiktoken

# DPO (Direct Preference Optimization) dataset
# key difference from SFT: we have pairs of responses (chosen vs rejected)
# we don't generate during training - we SCORE how likely each response is
# then train model to prefer chosen over rejected

# how many models in DPO?
# 1. policy model = the model we're training (SFT model + LoRA adapters being updated)
# 2. reference model = frozen SFT model (our anchor, prevents drift)
# they can SHARE base weights! policy has base(frozen) + adapters(training)

# what are log probabilities?
# NOT generation! we feed prompt+response through model and ask:
# "how likely is this SPECIFIC response?" (the response is from dataset, not generated)
# at each position, model predicts next token probability
# we sum log(probability of actual token) over response positions
# this gives us a score: higher = model thinks this response is more likely

class DPODataset(Dataset):
    # format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    
    # prompt: the instruction/question
    # chosen: the preferred response (human ranked higher)
    # rejected: the worse response (human ranked lower)
    
    # we tokenize prompt+chosen and prompt+rejected separately
    # model will score both (compute log probabilities)
    # loss makes chosen more likely, rejected less likely
    
    def __init__(self, data_path, max_length=512, tokenizer_name="cl100k_base"):
        self.max_length = max_length
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # load preference pairs from json
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} preference pairs for DPO")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # tokenize prompt+chosen sequence
        # this is what we'll score with the model
        # we're NOT generating - we're measuring "how likely is this specific sequence?"
        chosen_text = f"{prompt}{chosen}"
        chosen_tokens = self.tokenizer.encode(chosen_text)
        
        # tokenize prompt+rejected sequence
        rejected_text = f"{prompt}{rejected}"
        rejected_tokens = self.tokenizer.encode(rejected_text)
        
        # we need to know where prompt ends to create loss mask
        # just like SFT, we only calculate loss on response tokens, not prompt
        # why? because we care about response quality, not prompt prediction
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)
        
        # truncate if too long
        if len(chosen_tokens) > self.max_length:
            chosen_tokens = chosen_tokens[:self.max_length]
        if len(rejected_tokens) > self.max_length:
            rejected_tokens = rejected_tokens[:self.max_length]
        
        # pad to same length (for batching)
        chosen_seq_len = len(chosen_tokens)
        rejected_seq_len = len(rejected_tokens)
        
        if chosen_seq_len < self.max_length:
            chosen_tokens = chosen_tokens + [0] * (self.max_length - chosen_seq_len)
        if rejected_seq_len < self.max_length:
            rejected_tokens = rejected_tokens + [0] * (self.max_length - rejected_seq_len)
        
        # create loss masks
        # mask=0 for prompt tokens (ignore when calculating log probability)
        # mask=1 for response tokens (include in log probability)
        # this is EXACTLY like SFT masking
        chosen_mask = [0] * self.max_length
        for i in range(prompt_length, min(chosen_seq_len, self.max_length)):
            chosen_mask[i] = 1
        
        rejected_mask = [0] * self.max_length
        for i in range(prompt_length, min(rejected_seq_len, self.max_length)):
            rejected_mask[i] = 1
        
        # convert to tensors
        chosen_ids = torch.tensor(chosen_tokens, dtype=torch.long)
        rejected_ids = torch.tensor(rejected_tokens, dtype=torch.long)
        chosen_mask = torch.tensor(chosen_mask, dtype=torch.float)
        rejected_mask = torch.tensor(rejected_mask, dtype=torch.float)
        
        # we return BOTH sequences
        # during training:
        # 1. policy model scores chosen → gets log_prob_chosen
        # 2. policy model scores rejected → gets log_prob_rejected
        # 3. reference model does the same
        # 4. DPO loss: make (log_prob_chosen - log_prob_rejected) larger
        #    while staying close to reference model's preferences
        return chosen_ids, rejected_ids, chosen_mask, rejected_mask


class DPODataPreprocessor:
    """
    converts different formats to DPO format
    
    common sources:
    - human feedback data (ranked responses)
    - synthetic data (good vs bad responses)
    - reward model scores (filter best vs worst)
    """
    
    def preprocess_preference_data(self, input_path, output_path):
        """
        convert preference data to DPO format
        
        input formats handled:
        - anthropic HH format: conversations with rankings
        - reward model format: responses with scores
        """
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        formatted_data = []
        
        for item in raw_data:
            # handle different input formats
            if "chosen" in item and "rejected" in item:
                # already in correct format
                formatted_data.append({
                    "prompt": item.get("prompt", ""),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                })
            
            elif "responses" in item and "scores" in item:
                # multiple responses with scores - pick best and worst
                responses = item["responses"]
                scores = item["scores"]
                best_idx = scores.index(max(scores))
                worst_idx = scores.index(min(scores))
                
                formatted_data.append({
                    "prompt": item["prompt"],
                    "chosen": responses[best_idx],
                    "rejected": responses[worst_idx]
                })
        
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed {len(formatted_data)} preference pairs")
        print(f"Saved to {output_path}")
        
        if formatted_data:
            print("\nSample:")
            print(f"Prompt: {formatted_data[0]['prompt'][:80]}...")
            print(f"Chosen: {formatted_data[0]['chosen'][:80]}...")
            print(f"Rejected: {formatted_data[0]['rejected'][:80]}...")

