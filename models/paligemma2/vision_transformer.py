import torch
import torch.nn as nn
from embedding import Embedding
from vision_encoder import Encoder


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.vision_config.hidden_size

        self.embeddings = Embedding(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
