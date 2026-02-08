# sentencepiece tokenization used by llama
# language agnostic tokenization treating spaces as tokens

# topics to cover:
# - unigram language model tokenization
# - treating whitespace as special character (▁)
# - language agnostic (works for any unicode)
# - training sentencepiece model
# - encode_as_pieces() vs encode_as_ids()
# - why llama uses sentencepiece instead of bpe
# - comparing to huggingface llama tokenizer
# - handling multiple languages in same vocab

# OBJECTIVE: understand llama tokenization and unicode handling
# see how sentencepiece differs from byte level bpe
