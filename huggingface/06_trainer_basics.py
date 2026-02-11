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

print(f"Dataset: {dataset}")        
print(f"Tokenizer: {tokenizer}")
print(f"Model: {model}")

def tokenize_function(examples): # what is this function for even? how do we even pass function in another map method?
    # this function takes in a batch of examples and applies the tokenizer to the "text" field of each 
    # example, returning the tokenized output.
    # the tokenizer will convert the text into input_ids and attention_mask, which are necessary 
    # for training the language model.
    # the truncation=True argument ensures that the tokenized sequences do not exceed the maximum length 
    # of 512 tokens, which is important for efficient training and to prevent memory issues.
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"Tokenized Dataset: {tokenized}")


def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

labeled = tokenized.map(create_labels, batched=True) # 

print(f"Labeled Dataset: {labeled}")
print(f"Columns in Labeled Dataset: {labeled['train'].column_names}")
print(f"Example from Labeled Dataset: {labeled['train'][0]}")

labeled.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) # 
print(f"Formatted Labeled Dataset: {labeled}")
print(f"Example from Formatted Labeled Dataset: {labeled['train'][0]}")
train_dataset = labeled["train"].select(range(1000))
eval_dataset = labeled["validation"].select(range(1000))

print(f"Train Dataset: {train_dataset}")
print(f"Eval Dataset: {eval_dataset}")


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

print(f"Training Arguments: {training_args}")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print(f"Data Collator: {data_collator}")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
print(f"Trainer: {trainer}")

trainer.train()

print("Training complete!")



















