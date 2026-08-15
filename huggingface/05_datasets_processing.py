# preprocessing involves 
# tokenize text 
# add special tokens 
# handle padding truncation
# create labels 
# convert to tensors 
# batch 
# thats it !





from datasets import load_dataset 
from transformers import AutoTokenizer, DataCollatorForLanguageModeling 
import torch 

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


example=dataset["train"][0]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False, # dont pad yet do it during batching 
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"] # drop original text 
)
print(tokenized["train"][0])
print(f"Number of tokens: {len(tokenized['train'][0]['input_ids'])}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokenized['train'][0]['input_ids'])}")

# creat labels 
def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy() # copy input ids to labels 
    return examples

labeled = tokenized.map(create_labels, batched=True)
print(f"Input IDs: {labeled['train'][0]['input_ids']}")
print(f"Labels: {labeled['train'][0]['labels']}")


labeled.set_format(type="torch", columns=["input_ids", "labels"])
print(f"Input IDs (tensor): {labeled['train'][0]['input_ids']}")
print(f"Labels (tensor): {labeled['train'][0]['labels']}")

sample = labeled["train"][0]
input_ids = sample["input_ids"]
labels = sample["labels"]
print(f"Input IDs (tensor): {input_ids}")
print(f"Labels (tensor): {labels}")


collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # dont mask tokens for causal language modeling
)

mini_batch = labeled["train"][:4]
print(f"Mini batch input IDs: {[sample['input_ids'] for sample in mini_batch]}")
print(f"Mini batch labels: {[sample['labels'] for sample in mini_batch]}")

batch = collator(mini_batch)
print(f"Batched input IDs: {batch['input_ids']}")
print(f"Batched labels: {batch['labels']}")

# MLM means masked languagem modelling from bert 
