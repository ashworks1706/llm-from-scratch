# the whole story of dealing with RNNs and LSTMs is that we want to use temporal data or sequential data
# that is to deal with text, time series, audio etc where order MATTERSS


# instead of processing input all at once, we process it sequentially with memory 
#
#
# this is how RNN did it : 
# that is, take current input, combine with previous hidden state (memory), output prediction, pass hidden state to next step 
# we take same weights and we use it every step 
# like tanh(W_hh * h{t} + Wxh * x{t})



# but the problem with RNN is that gradient at step 1 needs to flow through all 100 steps 
# there fore in early layers if W_hh eigen values < 1 gradient vanishes exponentially 
# so it doesnt learn anything later, like context issues 
#
#







# that is why we have LSTM 
# instead of simple hidden state update i.e 
# h_t = tanh(W * [h_{t-1}, x_t])
# we use gates to control information flow :
# What to forget from memory
# what to add to memory 
# what to output from meory 
# so basically it has memory cells and traffic controllers 
# in traditional RNN: h_t = tanh(w * h_{t-1})
# multiplied by W and tanh' at every step -> vanishes over long sequences 
# LSTM does element wise multiplication without W matrix 
# so gradient flows through C_t via addition 
# C_t = f_t * C_{t-1} + i_t * C 
# its like resnet skip connections, highway for gradients 
# cell state update becomes =  C_t = f_t * C_{t-1} + i_t * C 
# the addition is the thing, even if i_t * C_t vanishes , f_t * c_{t-1} provides gradients 
#

# Components : 
# Cell state (C_t) : long term memory high way 
# runs through entire sequence 
# information can flow unchanged (like resnetskip!!!!!!!)

# Hidden state (h_t) : Short-term working memory 
# what we output at each step 


# Gates - netowrks with sigmoid 



# here h_{t-1} is previous hidden state (short term memory)
# C_{t-1} is previous cell state (long term memory)


# Forrget gate : what to remove from C_t 

# f_t = σ(W_f * [h_{t-1}, x_t] + b_f)
# σ = sigmoid (outputs 0 to 1)
# W_f -> weight matrix 
# h_{t-1} -> the RAM / Short term context 
# x_t -> current input 
# b_f -> bias vector 
# f_t decides s: "How much of C_{t-1} to keep?"
# if 0, forget, 1 keep, 0.5 keep half 


# Input gate + candidate : what to add to C_t
# i_t = i_t = σ(W_i * [h_{t-1}, x_t] + b_i)       
# "How much new info to add?"                                      

# C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
# "What new info to potentially add                                     

# Only add: i_t * C̃_t (modulated by gate                                        
# Example: "The cat is ___"                                                     
# "fluffy" → high i_t (important info, add to memory!)                        
# "very" → low i_t (filler word, don't store)                                 


# Step 3: Update Cell State                                                       
# C_t = f_t * C_{t-1} + i_t * C̃_                                                
#        └─ forget old   └─ add new                                              
# This is the KEY!                                                              
# Cell state updated by:                                                        
# - Forgetting some old memory                                                
# - Adding some new memory                



# Output gate : what to output from C_t 
# o_t = σ(W_o * [h_{t-1}, x_t] + b_o)
# h_t = o_t * tanh(C_t)


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W_xh: input to hidden 
        self.W_xh = nn.Linear(input_size, hidden_size)
        # W_hh : hiddne to hidden (recurrent connection)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
        h_new = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h_new

input_size = 10 
hidden_size = 20 
rnn_cell = RNNCell(input_size, hidden_size)

x = torch.randn(1, input_size)
h=  torch.zeros(1, hidden_size)

h_new = rnn_cell(x,h)

seq_len = 5
batch_size = 2 
sequence = torch.randn(seq_len, batch_size, input_size)
h = torch.zeros(batch_size, hidden_size)

for t in range(seq_len):
    x_t = sequence[t]
    h = rnn_cell(x_t, h)
    print(f"Step {t+1}: input {x_t.shape} -> hidden {h.shape}")

print(f"Final hidden state : {h.shape}\n")



class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, sequence):
        # seqlen, batch, input_size
        batch_size = sequence.size(1)
        h = torch.zeros(batch_size, self.hidden_size)

        for t in range(sequence.size(0)):
            h = self.rnn_cell(sequence[t], h)

        # use final hidden state for classification 
        output = self.fc(h)
        return output 

model = RNN(input_size=10, hidden_size= 20, output_size=3)
sequence = torch.randn(5,2,10) # seq len = 5, batch = 2, input = 10
output = model(sequence)



long_seq = torch.randn(100, 1, 10, requires_grad=True)
target = torch.Tensor([1])

model = RNN(input_size=10,hidden_size=20, output_size=3)
output = model(long_seq)
loss = F.cross_entropy(output, target)
loss.backward()
grad_magnitude = long_seq.grad[0].abs().mean().item()






































