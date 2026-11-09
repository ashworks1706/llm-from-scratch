# multimodal projector to connect vision encoder to language model
# just a linear layer to map dimensions

import torch
import torch.nn as nn


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_dim = config.vision_config.hidden_size  # e.g. 1152
        text_dim = config.text_config.hidden_size  # e.g. 2048
        
        # simple linear projection to bridge the two modalities
        self.linear = nn.Linear(vision_dim, text_dim, bias=True)
    
    def forward(self, image_features):
        # image_features: (batch, num_patches, vision_dim)
        # output: (batch, num_patches, text_dim)
        # now image tokens can be concatenated with text tokens
        return self.linear(image_features)
