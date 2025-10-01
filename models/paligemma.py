from typing import Dictt, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import image
import torch

IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD = [0.5,0.5,0.5]

def add_images_tokens_to_prompt(prefix_prompt: str, bos_token: Optional[str], image_seq_len: int, image_token: str):
    # this function adds the image tokens to the prompt
    # for example, if the prompt is "a cat", and the image_seq_len is 4, and the image_token is <image>, then the output will be "<bos> <image> <image> <image> <image> a cat"
    # if bos_token is None, then the output will be "<image> <image> <image> <image> a cat"
    if bos_token is not None:
        return f"{bos_token} " + " ".join([image_token]*image_seq_len) + f" {prefix_prompt}"
    else:
        return " ".join([image_token]*image_seq_len) + f" {prefix_prompt}"

def resize(image,size,resample,reducing_gap):
    # image resizing means changing the size of the image, it is needed when the input image size is different from the model input size
    # for example, the input image size is 512x512, but the model input size is 224x224, so we need to resize the image
    height,width = size
    # what does image.resize mean? since image is a PIL image, it means the new size of the image
    resized_image=image.resize((width,height),resample=resample, reducing_gap=reducing_gap) # PIL size is (width,height)
    return resized_image 

def rescale(image,scale, dtype):
    # image rescaling means changing the value of the image, it is needed when the input image value range is different from the model input value range
    # for example, the input image value range is [0,255], but the model input value range is [0,1], so we need to rescale the image value
    rescaled_image=image * scale
    rescaled_image=rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image,mean,std):
    # image normalization means changing the value of the image to have a mean of 0 and a standard deviation of 1
    # it is needed when the input image value range is different from the model input value range
    # for example, the input image value range is [0,1], but the model input value range is [-1,1], so we need to normalize the image value
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image=(image - mean) / std
    return normalized_image

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
        # object detection means identifying the location of objects in an image and drawing a bounding box around them
        # object segmentation means identifying the location of objects in an image and drawing a mask around them
        # this is done by adding extra tokens to the tokenizer
        # these tokens are used to represent the location and segmentation of objects in the image
        # for example, <loc0001> represents the location of the first object in the image, <seg001> represents the segmentation of the first object in the image
        # by adding these tokens to the tokenizer, we enable the model to understand and generate text that includes references to specific locations or segments within an image
        EXTRA_TOKENS=[f"<loc{i:04d}>" for i in range(1024)] # tokens are used for object detection
        EXTRA_TOKENS+=[f"<seg{i:03d}>" for i in range(128)] # tokens are used for object segmentation
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
        # padding is the process of adding extra tokens to the input sequence to make it the same length as the longest sequence in the batch
        # truncation is the process of removing tokens from the input sequence to make it the same length as the longest sequence in the batch
        # both padding and truncation are used to ensure that all input sequences in a batch have the same length, which is required for efficient processing by the model
        # for example, if the longest sequence in the batch has a length of 10, and another sequence has a length of 8, we can add 2 padding tokens to the end of the shorter sequence to make it the same length as the longest sequence
        # the difference between input ID and input embedding is that input ID is the index of the token in the vocabulary, while input embedding is the vector representation of the token, input ID is used to look up the input embedding in the embedding matrix and get the input embedding
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data