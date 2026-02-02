# LSTMs can also be used in classification tasks than just regression based (stock market)
# Here's how LSTMs work for text based classification


# we have our text data -> tokenize -> embedding -> now we have infromation rich vectors 
# now these vectors are passed as sequence to LSTM to get final hidden state 
# then we take the hidden state and pass it throguh the fully connected neural layer to get final postiive or negative score 
# we do cross entroppy loss, backprop through lstm AND embeddings so it's a joint training process 


# in the final layer of LSTM we have FC layer that we use for final prediction, why not hidden states? because the final layer has the summary of everything 
# embedding isj ust a weight matrix of learnable weights, we just index tokenids into the weight matrix 


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from collections import Counter 
import os 
import re 


# Custom embedding 
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)
    def forward(self, token_ids):
        return self.weight[token_ids]

# vocabulary 

class Vocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx= {"<PAD>": 0, "<UNK>":1} 
        # we use this because in embedding layer 0 vector will be all zeros for padding and unkown words will have their own vector 
        # therefore we reserve these indices and start actual words from index 2 
        self.idx2word = {0:"<PAD>", 1: "<UNK>"}
        # {"the":2, "a":3, "is":4, ...}
        # {"<PAD>":0, "<UNK>":1, "the":2, "a":3, "is":4, ...}
        # {0:"<PAD>", 1:"<UNK>", 2:"the", 3:"a", 4:"is", ...}

    def build_voab(self, texts):
        word_counts = Counter() # this will hold word frequencies, Counter() is a dict subclass for counting hashable objects,
        # hashable means it can be used as a key in a dictionary
        for text in texts:
            words = re.findall(r'\b\w\+\b', text.lower()) # this will extract words from text
            # \b is word boundary, \w+ is one or more word characters, re.I for ignore case    
            word_counts.update(words) 

        # get the most common words
        # Counter has a method most_common() that returns a list of the n most common elements and their counts from the most common to the least.
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for idx, (word,_) in enumerate(most_common start=2): 
            # start=2 because we have reserved 0 and 1 for <PAD> and <UNK>
            # idx, (word,_) unpacks the tuple returned by most_common
            # for example : 
            # most_common = [("the", 5000), ("a", 4000), ("is", 3000)]
            # then idx will be 2,3,4 and word will be "the", "a", "is"
            self.word2idx[word] = idx 
            self.idx2word[idx] = word 


    def encode(self, text):
        words = re.findall(r'\b\w+\b', text.lower()) # extract words from text
        return [self.word2idx.get(word,1) for word in words] # get the index of the word, if not found return 1 (<UNK>)

    def __len__(self): # ?
        return len(self.word2idx)



# now we load the actual dataset 

def load_imdb(data_dir = "./aclImdb"):
    def read_files(path):
        text =[]
        files = os.listdir(path)
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(path,file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        return texts 

    









