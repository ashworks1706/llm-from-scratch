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

    def _get_stats(self, word_tokens):
        # now we count all adjacent pair frequencies across ALL words 
        pairs = defaultdict(int)

        for word, freq in word_tokens.items():
            # split word into tokens by spaces 
            tokens = word.split()

            # count all adjacent pairs 
            for i in range(len(tokens) - 1 ):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] +=freq

        return pairs 

    # hold up, why are we using tuples specifically for these operations?
    # because tuples are hashable, they are the best for fast immutability (do not require change), lookup and 
    # memory allotment. THAT is why!

    def _merge_pair(self, word_tokens, pair):
        # now we merge specific pair in ALL teh words so far 
        new_word_tokens = {}
        bigram = ' '.join(pair) # this bascially creates the string representation of the pair, 
        # for example if pair is ('l', 'o'), bigram will be 'l o' 
        replacement = ''.join(pair) # "lo" (no space)


        for word, freq in word_tokens.items():
            # replace all occurences of the pair in this word 
            new_word = word.replace(bigram, replacement)
            new_word_tokens[new_word] = freq

        # what did we just do? 
        # we took all the words and replaced the most frequent pair with the new merged token, 
        # for example if the most frequent pair is ('l', 'o'), we replaced all occurences of 'l o' with 
        # 'lo', so "l o w </w>" becomes "lo w </w>", and "l o w e r </w>" becomes "lo w e r </w>", and we 
        # keep doing this until we reach the target vocab size, which means we have merged enough pairs to 
        # get a vocab size of 256, which includes the special tokens and the original characters, and the 
        # new merged tokens.
        
        # this way we are building up our vocab in a data driven way, based on the frequency of adjacent 
        # pairs in the corpus, which is a clever way to capture common subwords and words without having to manually 
        # define them, and it allows us to have a flexible vocab that can adapt to the specific language 
        # and domain of the text we are working with.

        return new_word_tokens

    def train(self, texts, num_merges = None):
        if num_merges is None:
            # we start with 256 bytes and our objective is to get to vocab size 
            num_merges = self.vocab_size - len(self.special_tokens) - 256

        corpus = ' '.join(texts) # we merge em all text 

        word_tokens = self._get_word_tokens(corpus) # we split into chars 

        for i in range(num_merges):
            pairs = self._get_stats(word_tokens)

            if not pairs:
                break
            
            best_pair = max(pairs, key= pairs.get)

            # merge the form paired 
            word_tokens = self._merge_pair(word_tokens, best_pair)
            self.merges.append(best_pair)

        print("Complete!")

        self._build_vocab()

    def _build_vocab(self):
        # build vocab from all unqiue tokens we saw during training 
        vocab = self.special_tokens.copy()
        vocab_id = len(vocab)

        # we then get those formed bytes and add them 
        for i in range(256):
            # here we are adding all the original characters as tokens in our vocab, so we can recognize them, 
            # and then we will add the merged tokens on top of that, so we have a complete vocab that 
            # includes the original characters, the special tokens, and the new merged tokens that we 
            # learned from the training process.
            token = chr(i)
            if token not in vocab:
                vocab[token] = vocab_id
                vocab_id += 1 # we inccrement ids since its dynamic how many words are tehre  

        # add all learned merge tokens 
        for token1, token2 in self.merges:
            merged = token1+ token2 
            if merged not in vocab:
                vocab[merged] = vocab_id
                vocab_id +=1 

        self.token_to_id = vocab 
        self.id_to_token = { v : k for k, v in vocab.items()}
        self.vocab = vocab 

    # You cannot use a tokenizer from one model (e.g., Llama) with 
    # the embeddings of another (e.g., GPT-4) because the ID numbers won't match

    def encode(self, text, add_special_tokens=True):
        # encode text using the rules we just made 
        
        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id['[BOS]'])

        tokens = list(text)

        for token1, token2 in self.merges:
            merged = token1 + token2
            new_tokens = [] 

            i =0 

            while i<len(tokens):
                # we check if the current token and the next token form the pair we want to merge, if they do, we 
                # add the merged token to the new tokens list and skip the next token, otherwise we just add the 
                # current token to the new tokens list and move on to the next token.
                if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i+1] == token2:
                    new_tokens.append(merged)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1

                tokens = new_tokens


        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['[UNK]'])
        if add_special_tokens:
            ids.append(self.token_to_id['[EOS]'])
        return ids


    def decode(self, ids):
        # decode ids back to text using the vocab 
        tokens  = [] 
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
                if self.special_tokens and self.id_to_token[id] in self.special_tokens:
                    continue
                tokens.append(self.id_to_token[id])

        return ''.join(tokens)




texts = [
    "hello world",]

tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(texts, num_merges=50)
test_text = "hello world"
encoded = tokenizer.encode(test_text)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)

