#  lets sidetrack a bit 
#  tiketoken is openai's production tokenzier, it's desinged to be fast
#  normally i build things from scratch since this was written in rust, i wanna do it too, but the objective right now is 
#  to learn with pretrained so we're going with that .


# we can not always decode a token ID back to exacto riginal text since teh spaces in vocabulary arre quite tricky 
# and tokenizer differ from each model to model 


import tiktoken 
tik= tiktoken.get_encoding("cl100k_base")
vocab_size = 256

# this one means we encode it WITH special tokens 
text = "Hello, how are you doing today? I hope you're doing well!" # this is the text we want to encode
token_ids = tik.encode(text) # we just encode it like that 
print("token_ids") # this is the list of token ids
print("Decoded text: ", tik.decode(token_ids)) # this is the decoded text
token_ids_no = tik.encode_ordinary(text) # WITHOUT special tokens 
print("Token IDs without special tokens: ", token_ids_no) # this is the list of token ids without special tokens
print("Decoded text without special tokens: ", tik.decode(token_ids_no)) # this is the decoded text without special tokens 

# now try token byte 
print("Decoded text from byte encoding: ", tik.decode_bytes(token_ids)) # this is the decoded text from byte encoding 




