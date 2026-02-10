# tokenizers are like the key parts of preprocessing data for LLMs 
# it has the power to affect model vocabulary size (how mayn unique "pieces" the model knows ), determine the sequence lenght, same text is differetn number of tokens depending on different schemes 
# and how well it handles rare/new words in terms of performance 


# historically, tokenizers handled things character level, splitting words in to each cars like array, word less and subword level, 
# subword level is the modern day tokenizer since it balances small vocabulary size, reasonable ssequecne legnths and ability ot handle unkwone rate words by breaking them into known pieces 


class Tokenizer:
    def __init__(self, vocab=None, special_tokens=None):

        # tokenizers are TRAINED, such that they know a bit of fixed vocabulary for nouns, verbs, etc. 
        # we also add [PAD] tokens to fill up unknown or short form 'don"t' type words 
        # such as [BOS] which means begining of seq, [EOS] end of seq, [UNK] unkown words, [CLS] classification for bert token, [SEP] used by bert again to seperate sentences 
        self.special_tokens = special_tokens or {"[PAD]", "[BOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]"}

        self.vocab = self.special_tokens.copy() if vocab is None else vocab 
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=True):
        ids = []
        if add_special_tokens:
            ids.append(self.vocab['[BOS]'])

        # now we convert each character 
        for char in text :
            if char in self.vocab:
                ids.append(self.vocab[char])
            else:
                ids.append(self.vocab['[UNK]'])

        # end with special tokens if nequested sometimess
        # since sometimes we want to encode text without special tokens for certain tasks such as language modeling where we predict next token 
        # other times we want to add special tokens for classification tasks where we need to mark the beginning and end of sentences
        if add_special_tokens:
            ids.append(self.vocab['[EOS]'])
        
        return ids 

    def decode(self, ids, skip_special_token=True):
        # now we just look up token ids and conver  them back to text 
        tokens =[]
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                # skip special tokens if not needed 
                if skip_spcial_token and token in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]
                    continue 
                tokens.append(token)


        return ''.join(tokens)























