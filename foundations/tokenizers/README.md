tokenizers library for understanding how text becomes numbers. while transformers includes tokenizers this focuses on the underlying mechanics of tokenization algorithms.

tokenization is the bridge between human language and model input. different tokenization strategies make different tradeoffs between vocabulary size and token sequence length.

bpe (byte pair encoding) is used by gpt models. starts with characters and iteratively merges most frequent pairs. covers training bpe from scratch and understanding merge rules.

wordpiece is used by bert. similar to bpe but uses likelihood instead of frequency. covers differences from bpe and why bert chose this approach.

sentencepiece is used by llama and t5. treats spaces as special characters and works directly on unicode. covers unigram language model tokenization.

tiktoken is openai fast bpe tokenizer. covers using it for gpt models and understanding encoding differences between gpt2 gpt3 and gpt4 tokenizers.

understanding tokenizers helps debug why models behave strangely on certain inputs and why vocabulary size matters for model architecture.
