# [CLS] -> start of sentence (classification)
# [SEP] -> sepeartor between sentence
# [UNC] -> unknown 
# [MASK] -> masked token for training 
# [PAD] -> padding token for short sequences 


import torch 
from transformers import AutoTokenizer 

tokenizer =AutoTokenizer.from_pretrained("distilbert-base-uncased")

text1 = "Hello world"
tokens1= tokenizer(text1, return_tensors ="pt") # return pytorch tensors 

print("Text 1: ", text1)
print("Tokens: ", tokens1["input_ids"])
print("Attention Mask:", tokens1["attention_mask"])


decoded = tokenizer.decode(tokens1["input_ids"][0])
print("Decoded Text: ", decoded)

texts=  ["Hello world", "How are you?", "I'm learning NLP" ]

batch_no_pad = tokenizer(texts, padding=False, return_tensors=None) # why return_tensors = None ? because we want to see the list of token ids without converting to tensors 
# since the sequences have different lengths, we cannot create a single tensor without padding. By setting return_tensors to None, we get a list of token ids for each sequence 
# without padding, allowing us to see the raw tokenization results for each individual text.
print("Batch without padding:")
print(batch_no_pad["input_ids"])
decoded_batch = [tokenizer.decode(ids) for ids in batch_no_pad["input_ids"]]
print("Decoded Batch without padding:")
for i, decoded_text in enumerate(decoded_batch):
    print(f"Text {i+1}: {decoded_text}")
batch_pad = tokenizer(texts, padding=True, return_tensors="pt") # padding to the longest sequence in the batch and return as pytorch tensors
print("Batch with padding:")
print(batch_pad["input_ids"])
decoded_batch_pad = [tokenizer.decode(ids) for ids in batch_pad["input_ids"]]
print("Decoded Batch with padding:")
for i, decoded_text in enumerate(decoded_batch_pad):
    print(f"Text {i+1}: {decoded_text}")


