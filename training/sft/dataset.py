import torch
from torch.utils.data import Dataset
import json
import tiktoken

class SFTDataset(Dataset):
    # The key difference from pretraining:
    # - Pretraining: learn from raw text (predict next token everywhere)
    # - SFT: learn from instruction-response pairs (only predict response tokens)
    
    # During training, we:
    # 1. Concatenate instruction + response into one sequence
    # 2. Only calculate loss on response tokens (mask instruction tokens)
    # 3. This teaches the model: "given instruction, generate this response"
    
    def __init__(self, data_path, max_seq_len, tokenizer_name="cl100k_base"):

        self.max_seq_len = max_seq_len
        
        # We use tiktoken for Llama3 models (same as pretraining)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Special tokens for formatting
        # Format: <instruction>text</instruction><response>text</response>
        self.instruction_start = "<instruction>"
        self.instruction_end = "</instruction>"
        self.response_start = "<response>"
        self.response_end = "</response>"
        
        print(f"Loaded {len(self.data)} instruction-response pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        response = item["response"]
        
        formatted_text = (
            f"{self.instruction_start}{instruction}{self.instruction_end}"
            f"{self.response_start}{response}{self.response_end}"
        )
        
        tokens = self.tokenizer.encode(formatted_text)
        
        # tokenize instruction separately to know where to start masking
        instruction_text = f"{self.instruction_start}{instruction}{self.instruction_end}{self.response_start}"
        instruction_tokens = self.tokenizer.encode(instruction_text)
        instruction_length = len(instruction_tokens)
        
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        seq_len = len(tokens)
        if seq_len < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - seq_len)
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        # we do all this string slicing because we have all we need in one sequence and we have to correctly
        # identify which is instruction and whihc is response of llm, since we want the llm to train on the response token 
        # theres no point of lelarning instruction tokens since llm provide answers 
        
        # loss mask: 0 for instruction tokens, 1 for response tokens
        # this is the key difference from pretraining
        loss_mask = [0] * len(input_ids)
        for i in range(instruction_length - 1, min(seq_len - 1, len(loss_mask))):
            loss_mask[i] = 1
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float)
        
        return input_ids, target_ids, loss_mask


class SFTDataPreprocessor:
   
    def __init__(self):
        pass
    
    def preprocess_conversations(self, input_path, output_path):
        # - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
        # - ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        formatted_data = []
        
        for item in raw_data:
            # Handle different input formats
            if "instruction" in item and "output" in item:
                # Alpaca format
                instruction = item["instruction"]
                if "input" in item and item["input"]:
                    instruction = f"{instruction}\n\nInput: {item['input']}"
                formatted_data.append({
                    "instruction": instruction,
                    "response": item["output"]
                })
            
            elif "conversations" in item:
                # ShareGPT format - extract first human-gpt pair
                conversations = item["conversations"]
                for i in range(len(conversations) - 1):
                    if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
                        formatted_data.append({
                            "instruction": conversations[i]["value"],
                            "response": conversations[i+1]["value"]
                        })
                        break
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed {len(formatted_data)} examples")
        print(f"Saved to {output_path}")
        
        if formatted_data:
            print("\nSample:")
            print(f"Instruction: {formatted_data[0]['instruction'][:100]}...")
            print(f"Response: {formatted_data[0]['response'][:100]}...")

