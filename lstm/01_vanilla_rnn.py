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

# Components : 
# Cell state (C_t) : long term memory high way 
# runs through entire sequence 
# information can flow unchanged (like resnetskip!!!!!!!)

# Hidden state (h_t) : Short-term working memory 
# what we output at each step 


# Gates - netowrks with sigmoid 
# Forrget gate : what to remove from C_t 
# Input gate : what to add to C_t
# Output gate : what to output from C_t 














