# BPE algorithm was a next step after subword tokenization 
# why? because it used a clever technique to form new tokens 
# the training process is basically - 
# start with every individual character present in the text as a separate token, 
# identify all adjacent pairs of tokens and count how often they appear in the corpus 
# select THAT pair and merge it with ONE NEW token, so now we're increasing our radar 
# and the new word is recognized for the reason that it's subwords are being used so much in the corpus, just like as humans we notice those things 


# in short, we coutn adjacent pairs, and keep track of them, the ones the most frequent, we merge it with the next biggest frequency tokne 
# and we keep repeating by merging most frequent cars until we reach the target vocab size, 
# for example if its 100, we do 100-13 = 87 more merges, we statistically solved this "the most manual thing" ever lol 









