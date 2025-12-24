# earlier in classification age of CNN/s, they look at images 
# through moving window kernels, neighboring pixels 
# by stacking many layers and slowly zooming out 
# but here in ViT, as we know with attention, we essentially 
# make the pixels make sense and talk to each other
# while CNN was a more of manual way of identifying things


import torch
 import torch.nn as nn 

 class SiglipVisionEmbeddings(nn.Module):
     def __init__(self, config):
         super().__init__()
         self.config = config
         self.embed_dim = config.vision_config.hidden_size
         self.image_size = config.vision_config.image_size
         self.patch_size = config.vision_config.patch_size
        

        # the first challenge is to convert the 2d image to 1d:
        # to do that, we use patching, we cut the image into patches
        # by using kernel (sliding window) and using projection matrix as the weights
        # for the linear projection, to avoid overlaps on images, 
        # we use window to caluclate the vector for the first patch
        # and then we jump to next patch with zero overlap
        # along with patching we pass the patch, through a 
        # linear layer to map the RGB to a vector sized D
        
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size= self.patch_size,
            stride= self.patch_size,
            padding="valid"
        )

        # now once we have patches, we need to keep track of which image
        # is placed where, for that we use Positional Embedding, 
        # that we apply on the patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        

