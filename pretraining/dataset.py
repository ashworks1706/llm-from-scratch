import torch
from torch.utils.data import Dataset
import tiktoken
class TextDataset(Dataset):
    def __init__(self, tokens, max_seq_len):
        #  Store tokens and max_seq_len
        self.tokens = tokens
        self.max_seq_len = max_seq_len
        #  Calculate how many samples you can create
        self.num_samples = (len(tokens) - 1) // max_seq_len
        # since the input is supposed to be sequentially split if its [1,2,3,4,5,7,8] and num max seq len is 4, we should be able to get 1 sample because 
        # we need 5 tokens for the first sample (4+1 for last target), leaving 3 tokens whihc isnt enough for antoher sample 
    
    def __len__(self):
        #  Return number of samples
        return self.num_samples
    
    def __getitem__(self, idx):
        # Extract the right chunk for this idx
        start_idx = idx * self.max_seq_len
        chunk = self.tokens[start_idx : start_idx + self.max_seq_len + 1]
        # split into input and target pairs
        input_tokens = chunk[:-1] # first maxseqlen tokens 
        target_tokens = chunk[1:] # shifted by 1
        # Return (input, target) pair

        return torch.tensor(input_tokens), torch.tensor(target_tokens)



class DataPreprocessor:
    def __init__(self, config):
        self.text_file_dir = config.text_file_dir

    def preprocess_text_file(self, file_path: str, output_path: str) -> None:
        # read the file path with graceful error handling 
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text:
                raise ValueError(f"File is empty!")
            # use tokenizer for tokenizing each word one by one 
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            # pay attention to max sequence length and add padding or truncation as required
            # save preprocessed tokens in input,target pairs in output dir 
            torch.save(tokens, output_path) # we save it in disk 
            # also retunr preprocess tokens 
            print(f"Preprocessed {file_path}")
            print(f"Total tokens: {len(tokens)}")
            print(f"First 20 tokens: {tokens[:20]}")
            print(f"Saved to: {output_path}") 

        except FileNotFoundError:
            print(f"Error: File not found!")
            raise
        except Exception as e:
            print(f"Error found {e}")
            raise
