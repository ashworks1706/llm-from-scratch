# using tiktoken for openai models
# fastest bpe implementation for production

# topics to cover:
# - tiktoken.encoding_for_model("gpt-4")
# - encode() and decode() methods
# - encode_ordinary() vs encode() (special tokens)
# - counting tokens before sending to api
# - differences between gpt2, gpt3.5, gpt4 encodings
# - handling special tokens explicitly
# - batch encoding for efficiency
# - why tiktoken is faster (rust implementation)

# OBJECTIVE: use tiktoken to count tokens and understand pricing
# see how different models tokenize same text differently
