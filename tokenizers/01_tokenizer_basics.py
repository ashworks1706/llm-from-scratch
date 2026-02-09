# tokenizers are like the key parts of preprocessing data for LLMs 
# it has the power to affect model vocabulary size (how mayn unique "pieces" the model knows ), determine the sequence lenght, same text is differetn number of tokens depending on different schemes 
# and how well it handles rare/new words in terms of performance 


# historically, tokenizers handled things character level, splitting words in to each cars like array, word less and subword level, 
# subword level is the modern day tokenizer since it balances small vocabulary size, reasonable ssequecne legnths and ability ot handle unkwone rate words by breaking them into known pieces 


# tokenizers are TRAINED, such that they know a bit of fixed vocabulary for nouns, verbs, etc. 
# we also add [PAD] tokens to fill up unknown or short form 'don"t' type words 
# such as [BOS] which means begining of seq, [EOS] end of seq, [UNK] unkown words, [CLS] classification for bert token, [SEP] used by bert again to seperate sentences 



