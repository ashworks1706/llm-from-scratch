# full paligemma model combining vision and language
# key idea: treat image patches as tokens and concatenate with text tokens

import torch
import torch.nn as nn
from vision_transformer import VisionTransformer
from gemma_decoder import GemmaDecoder
from projector import MultiModalProjector


class PaliGemma(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # three main components
        self.vision_tower = VisionTransformer(config)  # processes images into patch embeddings
        self.multi_modal_projector = MultiModalProjector(config)  # maps vision dim to text dim
        self.language_model = GemmaDecoder(config)  # generates text
        
        # special tokens
        self.pad_token_id = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.image_token_id = config.image_token_id if hasattr(config, 'image_token_id') else 256000
    
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        kv_cache_list=None,
        start_pos=0
    ):
        # process image if provided
        if pixel_values is not None:
            # vision encoder: (batch, 3, H, W) → (batch, num_patches, vision_dim)
            vision_outputs = self.vision_tower(pixel_values)
            
            # project to text space: (batch, num_patches, vision_dim) → (batch, num_patches, text_dim)
            image_features = self.multi_modal_projector(vision_outputs)
        else:
            image_features = None
        
        # process text
        if input_ids is not None:
            # get text embeddings
            text_embeddings = self.language_model.embed_tokens(input_ids)
            
            # if we have image features, concatenate them before text
            # this is the key: [image tokens, text tokens] as one sequence
            if image_features is not None:
                inputs_embeds = torch.cat([image_features, text_embeddings], dim=1)
            else:
                inputs_embeds = text_embeddings
        else:
            # image-only mode
            inputs_embeds = image_features
        
        # pass through language model layers
        # the self attention will let text tokens attend to image tokens naturally
        hidden_states = inputs_embeds
        for i, layer in enumerate(self.language_model.layers):
            layer_cache = kv_cache_list[i] if kv_cache_list is not None else None
            hidden_states = layer(hidden_states, layer_cache, start_pos)
        
        # final norm
        hidden_states = self.language_model.norm(hidden_states)
        
        # project to vocabulary to get logits
        # using weight tying: share embeddings with output projection
        logits = torch.matmul(
            hidden_states, 
            self.language_model.embed_tokens.weight.t()
        )
        
        return logits
    
    def merge_input_ids_with_image_features(self, image_features, input_ids):
        """
        Merge image features with text tokens at image token positions.
        
        In the input_ids, special image tokens mark where images should be inserted.
        This function replaces those special tokens with actual image features.
        """
        batch_size = input_ids.shape[0]
        
        # find positions of image tokens
        image_token_mask = input_ids == self.image_token_id
        
        # get text embeddings for all tokens
        text_embeddings = self.language_model.embed_tokens(input_ids)
        
        # replace image token positions with image features
        # this is simplified - real implementation needs to handle multiple images per sequence
        if image_token_mask.any():
            # for each sample in batch, insert image features at image token positions
            for batch_idx in range(batch_size):
                mask = image_token_mask[batch_idx]
                if mask.any():
                    # replace first image token with all image features
                    # this assumes one image per text for simplicity
                    first_image_token_pos = mask.nonzero()[0].item()
                    
                    # split text embeddings at image position
                    before = text_embeddings[batch_idx, :first_image_token_pos]
                    after = text_embeddings[batch_idx, first_image_token_pos+1:]
                    
                    # concatenate: [before, image_features, after]
                    text_embeddings[batch_idx] = torch.cat([
                        before,
                        image_features[batch_idx],
                        after
                    ], dim=0)
        
        return text_embeddings
