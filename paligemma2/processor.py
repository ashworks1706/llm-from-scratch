# processor for handling images and text inputs

import torch
from PIL import Image
import numpy as np


class PaliGemmaProcessor:
    """
    Processes images and text for PaliGemma model.
    
    Image processing:
    - Resize to 224x224
    - Normalize with ImageNet stats
    - Convert to tensor
    
    Text processing:
    - Tokenize text (simplified - would use real tokenizer in practice)
    - Add image tokens where needed
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # imagenet normalization
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def process_images(self, images):
        processed = []
        
        for img in images:
            # load if path
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            
            # resize
            img = img.resize((self.image_size, self.image_size))
            
            # to tensor and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
            
            # normalize
            img_tensor = (img_tensor - self.image_mean) / self.image_std
            
            processed.append(img_tensor)
        
        return torch.stack(processed)
    
    def process_text(self, texts, add_image_token=True, image_token_id=256000):
       
        # for now just showing the structure
        
        processed = []
        for text in texts:
            # in real implementation, tokenize properly
            # for now, just create dummy token ids
            tokens = [ord(c) % 1000 for c in text[:50]]  # dummy tokenization
            
            if add_image_token:
                tokens = [image_token_id] + tokens
            
            processed.append(tokens)
        
        # pad to same length
        max_len = max(len(t) for t in processed)
        padded = []
        for tokens in processed:
            padded_tokens = tokens + [0] * (max_len - len(tokens))
            padded.append(padded_tokens)
        
        return torch.tensor(padded, dtype=torch.long)
    
    def __call__(self, images=None, text=None):
        output = {}
        
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            output['pixel_values'] = self.process_images(images)
        
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            output['input_ids'] = self.process_text(text, add_image_token=images is not None)
        
        return output
