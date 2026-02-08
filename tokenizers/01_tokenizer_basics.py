# fundamental concepts of tokenization
# how and why we convert text to integers

# topics to cover:
# - what is a token (subword vs word vs character)
# - vocabulary and vocab size tradeoffs
# - encode() method (text to ids)
# - decode() method (ids back to text)
# - special tokens (CLS, SEP, PAD, UNK, BOS, EOS)
# - why we need padding and attention masks
# - handling unknown words
# - reversibility (can you always decode perfectly?)

# OBJECTIVE: understand why "unhappiness" might become ["un", "happiness"]
# see tradeoff between vocabulary size and sequence length
