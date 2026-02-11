# huggiingface hub is used for posting model cards online as well as datasets
# its like a github but for models and versions controllers, private public repos, etc 
# the models are usually posted in the repos as .bin 


import torch 
import torch.nn.functional as F 
from huggiingface_hub import login,model_info
from transformers import AutoModelForCasualLm, AutoTokenizer
from datasets import load_dataset

login() # for logging in 




model = AutoModelForCasualLm.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()

labeled = tokeizer.map(create_labels, batched=True)

labeled.set("torch", columns=["input_ids", "attention_mask"])
