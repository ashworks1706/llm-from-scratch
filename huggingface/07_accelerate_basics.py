# this is part of distrbuted training, how coudl u use more gpus instead of ur one single broke ass laptop gpu ?

# thats hwy we use Accelerate 

# we work with device management, distributed data parallele, mixed precision, gradient accumulation, gradient checkpointing 



# we DONT call .to(device ) in acclerated at all 

# gradinet checkpointing isj ust trade compute for memory, we dont store activatisn during forward pass 
# and we recompute during backward ppass, useful for very large models



# how to determine correct batch size given X gpus?
# effective batch size = batch_size * no. of GPUs
# local batch size = (total gpu mem / num of pars * 8 )
# which means that each gpu processes 32 examples in prallele that means 128 per time step 
#

# why do we even need gradinet accumulation?
# if we can only fit 32 batches per gpu and want effective batch of 128, we accumulate over 4 steps without increasing memory 



# why kee some fp32 in mixedp recision?
# because fp16 has low range and precision, loss vlaues can underflow or be infinity, so we keep some values in fp32 to maintain stability


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator

import torch 
from torch.optim import AdamW
from torch.utils.data import DataLoader


# data loader is for loading data in batches, we use it to load our dataset in batches for training while load_dataset is for loading the dataset itself 


# 1, load dataset and tokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples


labeled = tokenized.map(create_labels, batched=True)
labeled.set_format(type="torch", columns=["input_ids","attention", "labels"])

train_dataset = labeled["train"].select(range(500))
accelerator = Accelerator(mixed_precision="fp16")


optimizer = AdamW(model.parameters(), lr=5e-5)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # thhis is for language modeling, we dont use masked language modeling so mlm is False
# collate means to combine a list of samples into a batch, it handles padding and other necessary adjustments to create a batch of data that can be fed into the model 
# we use data collator to handle the batching and padding of our data, it ensures that all sequences in a batch are of the same length by padding them to the maximum length in the batch, and it also creates attention masks to indicate which tokens are padding and which are not

# steps -> epochs * (dataset size / batch size) = 3 * (500 / 32) = 47 steps per epoch, so total 141 steps for 3 epochs 
# but what are tehse terms like steps, batch, epochs, etc.
# batch size is the number of samples processed before the model is updated, it determines how many
# epochs is the number of complete passes through the training dataset, it determines how many times the model will see the entire dataset during training 
# steps is the number of batches processed during training, it is calculated as (dataset size / batch size) * epochs, it determines how many times the model will be updated during training 
# so in our case, with a batch size of 32 and a dataset size of 500, we will have 15.625 batches per epoch, which means we will have 47 steps per epoch, and with 3 epochs, we will have a total of 141 steps for training.

# in llms, first we have dataset, then we take that dataset and we split it into batches, these batches contain a certain rows of dataset,
# then we take that, in each epoch, we take each batch, and in each epoch one training step is one forward and backward pass of the model on one batch of data, so in each epoch.

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator) # collate_fn is the function that will be used to collate the samples into a batch, in this case we use the data collator we defined earlier to handle the batching and padding of our data
# but why do we even need collate_fn, why cant we just use the default collate function of the DataLoader?
# the default collate function of the DataLoader simply stacks the samples into a batch, it does not handle padding or attention masks, which are necessary for training language models, the data collator we defined earlier takes care of these necessary adjustments to create a 
# batch of data that can be fed into the model, it ensures that all sequences in a batch are of the same length by padding them to the maximum length in the batch
# , and it also creates attention masks to indicate which tokens are padding and which are not, so we need to use the collate_fn to ensure that our data is properly prepared for 
# training our language model.


model,optimizer, train_data_loader = accelerator.prepare(model, optimizer, train_data_loader)

print(f"Number of training steps: {len(train_data_loader) * 3}") # we have 3 epochs, so we multiply the number of batches in the train_data_loader by 3 to get the total number of training steps for 3 epochs
print(f"Number of batches per epoch: {len(train_data_loader)}") # we can get the number of batches per epoch by simply getting the length of the train_data_loader, which is the number of batches in our training dataset, since we have 3 epochs, we will have the same number of batches in each epoch, so we can just get the length of the train_data_loader to get the number of batches per epoch
print(f"Effective batch size: {32 * accelerator.num_processes}") # effective batch size is the batch size multiplied by the number of processes (GPUs) being used for training, so we multiply our batch size of 32 by the number of processes to get the effective batch size, which is the total number of samples processed in parallel across all GPUs during training
print(f"Local batch size: {32}") # local batch size is the batch size that each individual GPU processes, in this case we set our batch size to 32, so each GPU will process 32 samples in parallel during training, regardless of the number of GPUs being used, so the local batch size remains 32 even if we use multiple GPUs for training.



accelerator_with_accum = Accelerator(
    mixed_precision = 'fp16',
    gradient_accumulation_steps=4, # this basically means that we will accumulate gradients over 4 steps before updating the model, so we will effectively have a batch size of 32 * 4 = 128, which is the effective batch size we want to achieve, without increasing the 
   # memory requirements of our model, since we are only processing 32 samples at a time on each GPU, and we are accumulating the gradients over 4 steps before updating the model
)
