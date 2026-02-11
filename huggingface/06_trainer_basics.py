# the trainer class handles 
# Forward backward pass 
# logging to tensorboard
# evaluatiaon during training 
# learning rate scheduling 
# gradient accumulation
# mixed precision 



# we set warmup_steps = 500 because learning rate starts low and gradually increases 
# to full LR over 500 steps, rpevents training instability at the satrt 




from datasets import load_dataset 
from transformers import (
    AutoTokenizer,
    AutoModelForCasualLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch 

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCasualLM.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

labeled = tokenized.map(create_labels, batched=True) # 
labeled.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) # 

train_dataset = labeled["train"].select(range(1000))
eval_dataset = labeled["validation"].select(range(1000))

training_args = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 1, 
    learning_rate=5e6,
    per_device_train_batch_size=8,
    per_device_test_batch_size=8,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=50,
    warmup_steps=100,
    seed=42,
)






















