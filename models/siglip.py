import torch 
import torch.nn as nn 

class SiglipVisionConfig:

    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, num_channels=3, image_size=224, patch_size=16, layer_norm_eps=1e-6, attention_dropout=0.0, num_image_tokens: int=None, **kwargs):
        super().__init__()

        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.patch_size=patch_size
        self.image_size=image_size
        self.attention_dropout=attention_dropout
        self.layer_norm_eps=layer_norm_eps
        self.num_image_tokens=num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
       
        # this layer splits the input image into non overlapping patches and projects on high dimensional embedding space for best processing on transformer models 
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, # dimension of input chanenels
            out_channels=self.embed_dim, # '' output channels
            kernel_size=self.patch_size, # non overlapping patches of image will be processed
            stride=self.patch_size, # no overlap between patches on conv image layer
            padding="valid"
        ) 
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # add position embedding for denoting the parts of cnv layer
        self.register_buffer("position_ids",torch.arrange(self.num_positions).expand(1,-1), persistent=False)
        
    def forward(self,pixel_values: torch.FloatTensor):
        _,_,height,width = pixel_values.shape
        
        patch_embeds = self.patch_embedding(pixel_values)
        
        embeddings=patch_embeds.flatten(2)
        embeddings=embeddings.transpose(1,2)
        # add position embeddings to each patch as a vector
        
        embeddings= embeddings+self.position_embedding(self.position_ids)
        
        #  its not like we're telling the model this is embedding 1 positon 1, we are instead adding another embedding -> positional encoding that is added to the patch and then the model will learn how to modify the embedding in such a way that it matches the way it should encode the information through attention mechanisms 
        
        return embeddings 


class SiglipMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # intermediate size is mostly 4 times the hidden size, non linearity is added and compressed to hidden dimensions
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self,hidden_states):
        hidden_states = self.fc1(hidden_states)
        # theres no rule of thumb for non linearity functions -> flow of gradients
        # gelu [0,inf)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# batch norm is size sensitive so its necessary to have big batch size for good training

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self,hidden_states: torch.Tensor):
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states) # doesnt change shape but embeddding dimensions 
        hidden_states,_ = self.self_attn(hidden_state=hidden_states)
        hidden_states = residual+hidden_states
        
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states) # doesnt change shape but embedding dimensions
        
        hidden_states = self.mlp(hidden_states) # theres no mixing of batches its all independent

class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings= SiglipVisionEmbeddings(config)
        
        self.encoder= SiglipEncoder(config)
        # normalization is basically 
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds= hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values):
        # batch_size, num_channels, height, width = pixel_values.shape
        return self.vision_model(pixel_values=pixel_values)