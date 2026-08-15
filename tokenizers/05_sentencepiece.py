# we openai''s way, BPE -> character based -> merge characers into subwords
# now Meta brought something different
# they said, lets treat spaces as tokens (marked with special _ symbol)
# using this kind of unicode will work very nicely at low level since at lowest level we have unicode 
# so in that case we won't have to frequency count stuff 
# but how and why does this work? 
# the idea is that we can treat spaces as tokens, and then we can merge characters into subwords based on 
# their frequency in the training data.
# this way we can have a more efficient representation of the text, and we can also capture the 
# meaning of the text better since we can capture the subwords that are most relevant to the meaning of the text. 

# BPE (Bottom-Up): Starts with individual characters and merges them. 
# It is "greedy"—it keeps combining the most frequent pairs until the budget is hit.
# Unigram (Top-Down): Starts with a massive vocabulary (all possible substrings) and prunes them. 
# It calculates which tokens are "least useful" to the overall probability of the corpus and throws 
# them away until it hits the target vocab_size.

import sentencepiece as spm
import os 



def train(corpus_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix, 
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram', # unigram is a type of language model that is based on the assumption that the 
       #  probability of a word is independent of the words that come before it where as BPE is based on 
        #  the idea of merging the most frequent pairs of characters or subwords in the training data 
        #  until we reach the desired vocabulary size.
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        unk_piece='<unk>',
        pad_piece='<pad>',
        bos_piece='<s>',
        eos_piece='</s>'
    )

    return model_prefix

corpus_text = "This is a sample corpus for training the SentencePiece model. It contains multiple sentences to provide enough data for the model to learn from."

model_prefix = train(corpus_text, 'sentencepiece_model', vocab_size=1000)

def load_sentencepiece_model(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp


sp = load_sentencepiece_model(model_prefix)

# Example usage
text = "This is a test sentence to encode using the SentencePiece model."
encoded = sp.encode(text, out_type=str)
print("Encoded:", encoded)
decoded = sp.decode(encoded)
print("Decoded:", decoded)











