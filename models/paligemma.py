from typing import Dictt, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import image
import torch

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5,0.5,0.5]

class PaliGemmaProcessor:
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size:int):
        super().__init__()
        self.image_seq_length=num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS=[f"<loc{i:04d}>" for i in range(1024)] # tokens are used for object detection 
        EXTRA_TOKENS+=[f"<seg{i:03d}" for i in range(128)] # tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # we add BOS and EOS tokens
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
        
    
    def __call__(self, text, images, padding, truncation):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        
        pixel_values = process_images(images,size=(self.image_size, self.image_size), resample=Image.Resampling.BICUPIC, rescale_factor=1/255.0,image_mean=IMAGENET_STANDARD_MEAN, image_std=IMAGENET_STANDARD_STD)
        
        pixel_values = np.stack(pixel_values, axis=0)
        
        pixel_values = torch.tensor(pixel_values)