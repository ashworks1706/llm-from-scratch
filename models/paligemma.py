from typing import Dictt, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import image
import torch

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5,0.5,0.5]

def process_images(images,size,resample,rescale_factor,image_mean,image_std):
    height,width = size[0],size[1]
    images=[resize(image=image,size=(height,width),resample=resample) for image in images] # [0,1]
    images=[np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images] # have a mean 0 and SD 1
    images = [normalize(image,mean=image_mean,std=image_std) for image in images] # move channel dimension to first dimension. [channel,H,W]
    images=[image.transpose(2,0,1) for image in images]
    
    return images

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
        
        input_strings = [add_images_tokens_to_prompt(prefix_prompt = prompt, bos_token = self.tokenizer.bos_token, image_seq_len = self.image_seq_length, image_token=self.IMAGE_TOKEN) for prompt in text]
        
        inputs = self.tokenizer(input_strings, return_tensors="pt",padding=padding,truncation=truncation)
        
        return_data=