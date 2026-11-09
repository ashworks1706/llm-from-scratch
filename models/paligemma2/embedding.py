import torch
import torch.nn as nn 

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
    
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size= self.patch_size,
            stride= self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1,-1)), 
                             persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        embeddings = self.patch_embedding(pixel_values)
        embeddings = embeddings.flatten(2)
        embeddings = embeddings.transpose(1,2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


