import torch
import torch.nn as nn
from .decoder_block import DecoderBlock
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis 

class Llama(nn.Module): 
    def __init__(self, config):
        super().__init__()

        dim = config.embedding_size
        
        # 1. The Embedding Layer
        # Converts integer token IDs (e.g., 502) into dense vectors (size 512).
        # This is the "Lookup Table".
        self.tok_embeddings = nn.Embedding(config.vocab_size, dim)
        # torch.nn.Embedding is a trainable lookup table in PyTorch that converts discrete input indices into dense, 
        # low-dimensional vectors. 
        #  Instead of a high-dimensional one-hot vector, you input an integer index, and the layer outputs the 
        #  corresponding dense vector, making it computationally more efficient and effective for learning 
        #  from discrete data

        # 2. The Decoder Stack
        # We use nn.ModuleList, not a python list []. 
        # If we use a normal list, PyTorch won't register these layers for training.
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )

        # 3. Final Normalization
        # Llama normalizes the output of the last layer before making a prediction.
        # This stabilizes the final logits.
        self.norm = RMSNorm(dim, eps=config.rms_norm_eps)

        # 4. The Output Head (Language Modeling Head)
        # Projects the vector (512) back up to the vocabulary size (128k).
        # This gives us a score (logit) for every word in the dictionary.
        self.output = nn.Linear(dim, config.vocab_size, bias=False)

        # 5. Precompute RoPE Frequencies
        # We do this once at the start to save time.
        head_dim = dim // config.num_attention_heads
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=head_dim, 
            end=config.max_sequence_length * 2 # *2 is a safety buffer for context extension
        )

    def forward(self, tokens, kv_cache_list=None):
        # tokens shape: (Batch, Seq_Len)
        
        # 1. Convert IDs to Vectors
        h = self.tok_embeddings(tokens)

        # 2. Prepare RoPE frequencies
        seq_len = tokens.shape[1]
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len].to(tokens.device)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len].to(tokens.device)
        # we slice RoPE here specifically because, is freqs[:1] was there, we would rotate the 
        # second word after first word by 0 degrees
        # basically model thinks every single word is the first word in the sentence, hence output 
        # would be nonsense

        # 3. Run through the Decoder Stack
        for i, layer in enumerate(self.layers):
            # we must pick the correct cache object for this specific layer
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            # We pass the rotation frequencies to every layer
            h = layer(h, freqs_cos, freqs_sin, layer_cache, start_pos)

        # 4. Final Norm
        h = self.norm(h)

        # 5. Prediction
        # h shape: (Batch, Seq, Dim) -> logits shape: (Batch, Seq, Vocab_Size)
        logits = self.output(h)

        return logits
