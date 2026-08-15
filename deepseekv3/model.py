import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis
from .decoder_block import DeepSeekDecoderBlock

class DeepSeek(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dim = config.dim
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # We use ModuleList so PyTorch tracks all the parameters inside.
        self.layers = nn.ModuleList(
            [DeepSeekDecoderBlock(config) for _ in range(config.num_layers)]
        )
        
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Note: We use the 'rope_head_dim' specifically for rotation frequencies.
        # Unlike Llama which rotates the whole head, DeepSeek only rotates the 64-dim part.
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            dim=config.rope_head_dim, 
            end=config.max_seq_len * 2
        )

    def forward(self, tokens, start_pos=0, kv_cache_list=None):
        """
        tokens: (Batch, Seq)
        start_pos: Where are we in the sentence? (0 for prompt, N for generation)
        kv_cache_list: List of 32 Cache objects (one for each layer)
        """
        
        h = self.tok_embeddings(tokens)
        seq_len = tokens.shape[1]

        # B. Get RoPE Frequencies for THIS position
        # We slice the precomputed tables to get the angles for the current words.
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len].to(h.device)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len].to(h.device)

        # C. Run the Layers
        for i, layer in enumerate(self.layers):
            # Pick the correct cache for this layer
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            
            # Run the block
            h = layer(h, freqs_cos, freqs_sin, layer_cache, start_pos)

        # D. Output
        h = self.norm(h)
        logits = self.output(h)
        
        return logits
