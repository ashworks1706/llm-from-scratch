# we use datasets from this lirbary becausee it does batching inherently which is crucial since wee cant fit 
# entire dataset in ram and it also caches which is nice 


# dataset is single split, has many rows we can index 
# datadict is collection of named splots (train test val )
# if we set streaming=True, we dont download entire dataset, we stream from cloud 

from datasets import load_dataset 

dataset = load_dataset("imdb") # this is a datadict with 3 splits (train test val)
print(f"dataset splits: {dataset}")
print(f"dataset: {dataset['train']}") # this is a dataset with 25000 rows and 2 columns (text and label)
print(f"dataset row 0: {dataset['train'][0]}") # this is a dictionary with keys text and label 
print(f"dataset row 0 text: {dataset['train'][0]['text']}") # this is the text of the first row 
print(f"dataset row 0 label: {dataset['train'][0]['label']}") # this is the label of the first row (0 for negative, 1 for positive) 

# indexing and slicing works like lists and dictionaries
print(f"dataset rows 0-4: {dataset['train'][0:5]}") # this is a list of 5 dictionaries with keys text and label 
print(f"dataset rows 0, 2, 4: {dataset['train'][[0, 2, 4]]}") # this is a list of 3 dictionaries with keys text and label   



# filtering 
# we can filter the dataset based on a condition
def is_positive(example):
    return example['label'] == 1

positive_reviews = dataset['train'].filter(is_positive) # this will return a new dataset with only positive positive_reviews    
print(f"positive reviews: {positive_reviews}") # this is a dataset with only positive reviews
print(f"positive reviews row 0: {positive_reviews[0]}") # this is a dictionary with keys text and label, where label is 1 (positive)    

# mapping
# we can map a function to the dataset to transform the dataset
def add_length(example):
    example['length'] = len(example['text'])
    return example
dataset_with_length = dataset['train'].map(add_length) # this will return a new dataset with an additional column length which is the length of the text
print(f"dataset with length: {dataset_with_length}") # this is a dataset with an additional column length 
print(f"dataset with length row 0: {dataset_with_length[0]}") # this is a dictionary with keys text, label and length, where length is the length of the text of the first row  

