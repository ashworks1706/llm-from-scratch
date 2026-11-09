import torch.nn as nn
from vision_encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.vision_config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states
