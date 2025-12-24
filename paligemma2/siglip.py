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
    
    # we register the buffer for position_ids so they are saved with the model but not 
    # updated by the optimizer
    # but why? In a buffer, you store things the optimizer will NOT update
    # (like position_ids). Things the optimizer will update (like 
    # weights/biases) are nn.Parameter.
    
    self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1,-1)), 
                         persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # pixel_values shape : (Batch_size, 3 , height, width)
        
        # we apply patching
        # output shape: (batch, embed_dim, grid_h, grid_w)
        embeddings = self.patch_embedding(pixel_values)
        
        # Flatten them into sequence
        # shape: (batch, embed_dim, num_patches)
        embeddings = embeddings.flatten(2)
        
        # we transpose it to channel list (rules)
        # shape: (batch, 256, 1152)
        embeddings = embeddings.transpose(1,2)
        

        # adding position_embedding
        # we just add the learned vector for position 0 to patch 0,etc..
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipAttention(nn.Module):
    

    # just like normal attention we're dealing with image patches here, since we're encoding 
    # images no text
    
    # in a traditional attention context:
    # Q -> what is patch A looking for?
    # K -> what does patch B contain?
    # V -> what information should Patch B pass to Patch A if they match?
    
    # Attention(Q,K,V) = softmax(QK^T / sqrt(D_k)) * V
    # here, dot product (QK^T) measures similarity, such that Query and Key allignment will result in score being high
    # which creates 256*256 matrix representing how much every patch cares about other patch
    # we divide it by the square root of the head dimension to perform scaling to prevent dot products become too 
    # huge (vanishing gradient cause in softmax later on )
    # we then take a weighted sum of the values based on the probabilities and multiply it with the softmaxed score


    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_config.hidden_size # 1152
        self.num_heads = config.vision_conffig.num_attention_heads # 16
        self.head_dim = self.embed_dim // self.num_heads # 1152/ 16 = 72
        self.scale = self.head_dim ** -0.5 # 1 / sqrt(72)

        # projections (linear layers)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, embed_dim) -> (B,256,1152)
        batch_size, seq_len, _ = hidden_states.size()

        # Projection to Q, K, V 
        # shape: (B, 256, 1152)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # split it into heads 
        # we reshape 1152 -> 16 heads x 72 dimension
        # then we transpose to bring 'num_heads' before 'seq_len' so we can do 
        # matrix multiplication per head
        # target shape : (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # K transposed shape : (Batch, heads, head_dim, seq_len)
        # Result shape: (Batch, heads, seq_len, seq_len) -> (B,16,256,256)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        # we then use softmax, to convert all scores to probabilities laon the last dimension (columns)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)

        # apply to values
        # (B, 16, 256, 256) * (B, 16, 256, 72) -> (B, 16, 256, 72)
        attn_output = torch.matmul(attn_weights, value_states)

        # we then transpose them back (B, 256, 16, 72)
        attn_output=attn_output.transpose(1,2).contiguous()
        # flatte back 
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # final linear projection
        attn_output = self.out_proj(attn_output)

        return attn_output




