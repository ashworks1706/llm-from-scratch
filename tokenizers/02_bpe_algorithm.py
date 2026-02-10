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





from collections import defaultdict, Counter 
import re 

class BPETokenizer:
    def __init__(self, vocab_size=256, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            '[PAD]' : 0,
            '[UNK]' : 1,
            '[BOS]' : 2,
            '[EOS]' : 3,
        }

        # the will store merge rules (token1, token2) -> merge token 
        # order matters since we applly merges in the order they are discovered
        #
        self.merges =[]

        self.vocab={}
        self.id_to_token={}
        self.token_to_id={}

    def _get_word_tokens(self,text):
        # we first split into words 
        words = text.split()

        # count words frequency 
        word_freq = Counter(words)

        word_tokens = {}

        # split each word into chars with </vw> marker at the end, 
        # why? because we want to distinguish between "low" and "lower", if we just split into chars, 
        # we get l o w e r, but with the marker, we get l o w </w> and l o w e r </w>, so we can 
        # merge l and o to get lo, and then merge lo and w to get low, but low will only be merged if 
        # it appears frequently enough in the corpus, otherwise it will stay as l o w </w>, and we can 
        # still recognize low as a word because of the </w> marker, which indicates that it's the 
        # end of a word.
        for word,freq in word_freq.items():
            # low -> l o w </w>
            chars = ' '.join(list(word)) + ' </w>'
            word_tokens[chars] = freq 

        return word_tokens



