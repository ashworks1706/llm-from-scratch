# gemma decoder stack
# full language model that generates text

import torch.nn as nn
from gemma_rmsnorm import RMSNorm
from gemma_decoder_layer import GemmaDecoderLayer


class GemmaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.text_config.hidden_size
        
        # token embeddings: converts token ids to vectors
        self.embed_tokens = nn.Embedding(config.text_config.vocab_size, self.dim)
        
        # stack of decoder layers
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config) for _ in range(config.text_config.num_hidden_layers)
        ])
        
        # final normalization before output
        self.norm = RMSNorm(self.dim, eps=config.text_config.rms_norm_eps)

    def forward(self, input_ids, kv_cache_list=None, start_pos=0):
        # convert token ids to embeddings
        h = self.embed_tokens(input_ids)
        
        # pass through all decoder layers
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            h = layer(h, layer_cache, start_pos)
        
        # final normalization
        h = self.norm(h)
        return h
