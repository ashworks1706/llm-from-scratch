# but hey, not all pretrained methods are enough, 
# eventtually there are circumstances where we WANT a pretrained tokenzeir and we DONT
# we dont when we're training a domain specific model, working with a new language, want optimal compression for OUR data 
# then we train our own 


# so we take the tiktokenizer + huggignface's tokeniers library 




# so we just load the corpus, use tokenizer() + bpetrainer() from huggingface 
# train on corpus with target vocab size
# save for later use 





from tokenizers import Tokenizer 
from tokenizers.models import BPE 
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelDecoder 
from tokenizers.decoders import ByteLevel as ByteLevelDecoder 
from tokenizers.trainers import BpeTrainer


def train_tokenizer(corpus_texts, vocab_size=1000, output_path="custom_tokenizer.json"):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]")) # initialize tokenizer with BPE model 
    # train with bpe 
    trainer = BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[CLS]", "[SEP]"],
        show_progress=True,
    )

    # train on corpus 
    tokenizer.train_from_iterator(corpus_texts, trainer)

    # save it 
    tokenizer.save(output_path)

    return tokenizer


corpus = [
    "This is the first sentence.",
    "This is the second sentence.", 
    "And this is the third sentence."
]

tokenizer = train_tokenizer(corpus, vocab_size=50, output_path="custom_tokenizer.json")


# now lets test 
encoded = tokenizer.encode("This is the first sentence.")
print("Encoded:", encoded.tokens)













