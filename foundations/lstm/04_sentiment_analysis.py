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
        
        for idx, (word,_) in enumerate(most_common, start=2): 
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
    def read_files(path): # why is this defined in a function? 
        # because it's only used here 
        text =[]
        files = os.listdir(path)
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(path,file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        return texts 

    train_pos = read_files(os.path.join(data_dir,  'train/pos'))
    train_neg = read_files(os.path.join(data_dir,  'train/neg'))
    test_pos = read_files(os.path.join(data_dir,  'train/pos'))
    test_neg = read_files(os.path.join(data_dir,  'train/neg'))

    #  :
    # here we are trying to read all the positive and negative reviews from the IMDB dataset
    # we have a function read_files that takes a path and reads all the .txt files in that path and returns a list of texts
    # we call this function for train/pos, train/neg, test/pos, test/neg to get the respective reviews

    train_texts = train_pos + train_neg # here we combine positive and negative reviews 
    train_labels = [1] * len(train_pos) + [0] * len(train_neg) # 1 for positive, 0 for negative

    test_texts = text_pos + text_neg # combine positive and negative reviews

    test_labels = [1] * len(test_pos) + [0] * len(test_neg) # 1 for positive, 0 for negative


    return train_texts, train_labels, test_texts, test_labels 


# add paddings for irregular sequences on shape errors 
def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
            # this equation means we add 0s to the end of the sequence until it reaches max_len
            # for example if seq = [2,3,4] and max_len = 5
            # then we add [0,0] to the end to make it [2,3,4,0,0]
        else:
            padded.append(seq[:max_len])
    return padded 




class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size) # since we have 4 splits 
    def forward(self, x, h, C):
        combined = torch.cat([x,h], dim=1)
        # x is input 
        # h is previous vector 
        # C is new info 
        gates = self.W(combined)

        f,i,o,c_tilde = gates.chunk(4,dim=1)

        f = torch.sigmoid(f) # forget gate 
        i = torch.sigmoid(i) # input gate 
        o = torch.sigmoid(o) # ouput gate 

        c_tilde = torch.tanh(c_tilde) # new info 

        C_new = f * C + i * c_tilde # update the cell 

        h = o * torch.tanh(C_new) 
        return h, C_new 


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size 
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm_cell = LSTMCell(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #  :
        # here we are trying to define the forward pass of the LSTM model for sentiment analysis
        # x is the input tensor of shape (batch_size, seq_len)
        # and then we get the batch size and sequence length from the input tensor
        # then we pass the input through the embedding layer to get the embedded representation of shape (batch_size, seq_len, embedding_dim)
        # we initialize the hidden state h and cell state C to zeros
        # then we iterate over each time step in the sequence length
        # at each time step we pass the embedded input at that time step, along with the
        # previous hidden state and cell state to the LSTM cell to get the new hidden state and cell state
        # after processing the entire sequence, we pass the final hidden state through the fully connected layer
        # to get the output logits for the sentiment classes
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        C = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            h, C = self.lstm_cell(embedded[:,t,:], h, C)

        out = self.fc(h)
        return out 


train_texts, train_labels, test_texts, test_labels = load_imdb()
print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

vocab = Vocabulary(max_vocab_size= 10000)

vocab.build_vocab(train_texts)

print(f"Vocab size: {len(vocab)}")


# encoding voab 
max_len = 256
train_encoded = [vocab.encode(text) for text in train_texts] # ? 
train_padded = pad_sequences(train_encoded, max_len) # ?
train_encoded = [vocab.encode(text) for text in test_texts] # ? 
test_padded = pad_sequences(test_encoded, max_len ) # ? 

# here we are converting the text data into numerical format that can be fed into the model
# we first encode the training texts using the vocabulary to get a list of token ids for each review
# then we pad the encoded sequences to a fixed length of max_len using the pad_sequences function
# we repeat the same process for the test texts
# finally we convert the padded sequences and labels into PyTorch tensors for training and testing


# train test split basically means we divide the data into two parts, one for training the model and one for testing its performance
X_train = torch.LongTensor(train_padded) # ? 
y_train = torch.LongTensor(train_labels) #? 
X_test = torch.LongTensor(test_padded) #? 
Y_test = torch.LongTensor(test_labels) # ?


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device : {device}")



model = LSTM(vocab_size=len(vocab), embedding_dim = 128, hidden_size = 256, num_classes = 2).to(device)

criterion = nn.CrossEntropyLoss() # criterion means loss function
# why call this criterion? because it's a common term in machine learning for loss functions
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss =0 
    correct = 0
    total =0 

    for i in  range(0, len(X_train), batch_size):
        # first, we set the model to training mode using model.train()
        # then we initialize total_loss, correct, and total to keep track of the loss and accuracy
        # we iterate over the training data in batches of size batch_size
        # for each batch, we extract the input data batch_X and the corresponding labels batch_y
        # we move the data to the specified device (CPU or GPU)
        # we update the count of correct predictions and the total number of samples
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        # we zero the gradients of the optimizer to prepare for backpropagation
        optimizer.zero_grad()

       # we perform a forward pass through the model to get the output logits
       
        output = model(batch_X)
        # we compute the loss using the criterion (cross-entropy loss)
        
        loss = criterion(outputs, batch_y)
        # we perform backpropagation to compute the gradients
        loss.backward()
        # we update the model parameters using the optimizer
        optimizer.step()

        # we accumulate the total loss for the epoch
        total_loss += loss.item()

        # we calculate the predicted labels by taking the argmax of the output logits
        pred = outputs.argmax(dim=1)

        correct += (pred == batch_y).sum().item()

        total+= batch_y.size(0)

    acc = correct /total 

    print(f"epoch : {epoch+1/num_epochs}, loss : {total_loss/(len(X_train)//batch_size):4.f}, Acc: {acc:.4f}")


model.eval()
correct=0
total=0

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i+batch_size].to(device) # this is getting a batch of test data 
        batch_y = y_test[i:i+batch_size].to(device) # this is getting the corresponding labels for the test data 

        outputs = model(batch_X)
        pred = outputs.argmax(dim=1) # argmax gets the index of the max value along the specified dimension, why do we need this?
        # because the output of the model is a tensor of shape (batch_size, num_classes) containing the logits for each class
        # we need to convert these logits into predicted class labels by taking the index of the maximum logit for each sample in the batch
        correct+=(pred==batch_y).sum().item()  
        total+=batch_y.size(0) # ?


print(f"test acc : {correct/total:.4f}")


test_reviews = [
    "This movie was absolutely fantastic! Best film ever.",
    "Such a  shit ass movie bro, boring and poorly acted"
]


for review in test_reviews:
    tokens= vocab.encode(review)
    tokens_padded = pad_sequences([tokens], max_len)[0]
    x = torch.LongTensor([tokens_padded]).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    print(f"{"Positive" if pred ==1 else "Negative"}")
















