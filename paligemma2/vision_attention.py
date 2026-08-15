
import torch
import torch.nn as nn 

class Attention(nn.Module):
    

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



    # | **Feature**          | **Text Transformer (LLM)** | **Vision Transformer (ViT)**
    # |----------------------|----------------------------|------------------------------
    # | Input Data           | Token IDs (Integers)       | Raw Pixels (Floats)          
    # | Embedding Method     | `nn.Embedding` (Lookup Table) | `nn.Conv2d` (Math Projection) 
    # | Sequence Length      | Variable (sentences differ in length) | Fixed (256 patches for 224 px image) 
    

    # also in normal text based bert, we add CLS token at the start to represent the whole sentence 
    # but in paligemma, we don't, since we don't want to compress the image into single vector but want llm to see all 256 patches
    # for efficient attention mechanism


    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.vision_config.hidden_size # 1152
        self.num_heads = config.vision_config.num_attention_heads # 16
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
        
        # we do this because, pytorch's matrix mulitplication operates on the last two dimensions
        # since it treats everything else as a batch dimension to loop over 
        # if we did not do this, then it would matrix mulitply 16x72 which is wrong other than seq_len
        # by doing this, we tell pytorch to treat (batch,16) as independent matrices that it needs to process
        # Reshape : Split the big vector into heads
        # Transpose : Move 'Heads' dimension out of the way
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # K transposed shape : (Batch, heads, head_dim, seq_len)
        # Result shape: (Batch, heads, seq_len, seq_len) -> (B,16,256,256)
        # why? we do this to compress the feature dimension to find the relationship between sequence positions
        # if we did not do this, we would try to multiply 256x72 * 256x72, which is impossible
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        # we then use softmax, to convert all scores to probabilities laon the last dimension (columns)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)

        # apply to values
        # (B, 16, 256, 256) * (B, 16, 256, 72) -> (B, 16, 256, 72)
        attn_output = torch.matmul(attn_weights, value_states)

        # we then transpose them back (B, 256, 16, 72)
        attn_output=attn_output.transpose(1,2).contiguous() # <- this does not actually move data in memoory, it just changes
        # the stride to read the memory
        # now in RAM, the data for "head 1" is still far away from "head 0". it's fragmented.
        # flatten back 
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim) # <- this forces the pytroch to make a fresh copy 
        # of tensor where the memory layout physically matches the shape. it defragged the tensor so we can flatten it safely


        # final linear projection
        attn_output = self.out_proj(attn_output)

        return attn_output




