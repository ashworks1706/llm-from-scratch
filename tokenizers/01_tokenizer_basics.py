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

    def build_vocab(self,texts):
        # this is a short function for scanning all nique chars and assign them IDs, this is essentially the part to 'build' the vocabulary 
        next_id = len(self.vocab)

        unique_chars = set()

        for text in texts:
            unique_chars.update(text)

        # add each char to vocab 
        for char in sorted(unique_chars):
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id+=1

        # now we do reverse mappng 
        self.id_to_token = { v:k for k,v in self.vocab.item()} # this basicaly creates a reverse mapping from id to token for decoding 


    def pad(self, ids_list, max_length=None, pad_id=0):
        # we pad sequnces because NNs expect fixed size for matrices 
        if max_length is None:
            max_length = max(len(ids) for ids in ids_list)

        padded = []
        masks = []

        for ids in ids_list:
            # pad with [pad] token 
            padded_ids = ids + [pad_id] * (max_length - len(ids)) # this basically takes the original ids and appends enough pad_id to reach the max_length, if the original ids is 
            # already longer than max_length, it will just add zero pads which will be truncated in the next line 
            padded.append(padded_ids[:max_length]) #  this truncates the padded ids to max_length if it exceeds, ensuring all sequences are the same length for batch processing

            # create attentiion mask: 1 where real tokens, 0 where padded 
            mask = [1] * len(ids) + [0] * (max_length - len(ids)) # this creates a mask that has 1s for the original tokens and 0s for the padded tokens, which is used by the model to ignore the padded parts during training or inference
            masks.append(mask[:max_length])

        # now for example it would return, 
        # padded = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
        # masks = [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
        # where the first sequence has 3 real tokens and 2 pads, and the second sequence has 2 real tokens and 3 pads, and the masks indicate which are real tokens vs pads 

        return padded, masks 






















