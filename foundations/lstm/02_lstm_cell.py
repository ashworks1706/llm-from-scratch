import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # all gates combined in one linear layer 
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, C_prev):
        # combine input and previous hidden 
        combined = torch.cat([x, h_prev], dim=1) # batch, input_size + hidden_size
        # compute all gates at once 
        gates = self.W(combined) # batch, 4*hidden_size

        # splitting into 4 gates 
        # basically istead of 4 lienar layers we just use one and split since its all concatenation 
        forget_gate, input_gate, output_gate, candidate = gates.chunk(4, dim=1)

        # activations 
        f_t = torch.sigmoid(forget_gate) # 0 to 1 : howmuch to forget 
        i_t = torch.sigmoid(input_gate) # 0 to 1: how much to add 
        o_t = torch.sigmoid(output_gate) # 0 to 1 : how much to output 
        C_tilde = torch.tanh(candidate) # -1 to 1: candidate new info 
        # c_tilde is the new info that needs to be added 
        # i_t is HOW MUCH of the new info to add 


        # update cell state (long term memory)
        C_new = f_t * C_prev + i_t * C_tilde
        #       |- forget old   |- add new 

        # update hidden state (short term working memory )
        h_new = o_t * torch.tanh(C_new)

        return h_new, C_new


input_size = 10 
hidden_size = 20 
lstm_cell = LSTMCell(input_size, hidden_size)

x = torch.randn(2, input_size)
h = torch.zeros(2, hidden_size)
C = torch.zeros(2, hidden_size)
h_new , C_new = lstm_cell(x,h,C)






# Does lstm suffer from vanishing gradient ?
# No, LSTMs are designed to mitigate the vanishing gradient problem through their 
# gating mechanisms, which help maintain long-term dependencies in the data.
# However, they can still be affected by it in very deep networks or with very 
# long sequences, but generally, they perform much better than traditional RNNs 
# in this regard.
#
# why not use residual connections in lstm?
# While residual connections can be beneficial in deep networks to help with gradient flow, LSTMs already have 
# mechanisms (like gates) that help manage information flow over time. Adding residual connections could
# complicate the architecture without significant benefits, but it can be experimented 
# with in specific scenarios.
#
# Can we use batch normalization in LSTMs?
# Yes, batch normalization can be applied to LSTMs, but it is less common than in 
# feedforward networks. It can help stabilize training and improve convergence, but care
# must be taken to apply it correctly, typically on the inputs to the LSTM or on the 
# outputs of the LSTM layers.
#
# What are some alternatives to LSTMs for sequence modeling?
# Alternatives to LSTMs include Gated Recurrent Units (GRUs), Transformer models -> that do use residual connections extensively, 
# Temporal Convolutional Networks (TCNs), and simple RNNs with attention mechanisms. 
# Each has its own advantages and trade-offs depending on the specific application 
# and dataset.



















