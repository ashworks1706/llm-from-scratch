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

class SiglipAttention(nn.Module):
    
    # this is MHA from Attention is all you need paper.
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self,hidden_states):
        # hidden states : batchsize, numpatches, embeddimension
        batch_size,seq_len,_ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # QKV is just transformation of input sequence
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # each matrix has 8 elements -> heads, each element has 4 tokens, each token has 128 dimensions
        
        # attention = Q* K^T/sqrt(d_k). attnweights : batchsize, numheads, numpatches, numpatches
        
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)
        
        


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

# difference between VLM and LLM is that in VLM we want to contexualize the visuals and patches in such a way that they capture information about all other patches while in LLMs we want patches to catch informations about its token and previous tokens

# batch norm is size sensitive so its necessary to have big batch size for good training
# in attention mechanism, the model doesn't learn one word during one pass but all the loss gradients for the position and label in parallel in one pass, which is why its so powerful
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