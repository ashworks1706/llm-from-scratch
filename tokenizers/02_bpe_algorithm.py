# byte pair encoding from scratch
# understanding how gpt tokenizers work

# topics to cover:
# - starting with character level vocabulary
# - counting adjacent token pairs in corpus
# - merging most frequent pair
# - iterative process until desired vocab size
# - encoding new text using learned merges
# - why bpe handles rare words better than word level
# - byte level bpe vs character level bpe
# - train_new_from_iterator() to train your own

# OBJECTIVE: implement basic bpe algorithm and see how merges are learned
# train tokenizer on small corpus and see merge rules
