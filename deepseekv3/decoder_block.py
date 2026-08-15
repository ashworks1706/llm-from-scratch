# the depseek decoder has 
# input x
# layer 1 attention: Norm -> MLA -> Residual Add
# layer 2 feedforward: Norm -> MoE -> Residual Add


import torch
import torch.nn as nn
from .attention import MLA          # The new Memory efficient attention
from .moe import SparseMoE          # The 64-Expert Router (Fine-Grained)
from .rmsnorm import RMSNorm        # Standard normalization

class DeepSeekDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim = config.dim
        eps = 1e-6 # Standard epsilon for stability

        # DEPARTMENT 1: COMMUNICATION (MLA)
        # "The Efficient Memory Department"
        # 1. We normalize the data BEFORE it enters the department.
        self.attention_norm = RMSNorm(dim, eps=eps)
        
        # 2. The MLA Layer
        # This handles the Compression -> Storage -> Extraction logic we just discussed.
        self.attention = MLA(config)

        # DEPARTMENT 2: PROCESSING (MoE)
        # "The Specialized Thinking Department"
        # 1. We normalize the data again before processing.
        self.ffn_norm = RMSNorm(dim, eps=eps)
        
        # 2. The MoE Layer
        # Instead of 1 MLP, this routes to 6 of 64 Experts.
        # (Note: We use the same SparseMoE class, but config has num_experts=64)
        self.feed_forward = SparseMoE(config)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None, start_pos=None):
        """
        x: Input tensor (Batch, Seq, Dim)
        freqs_cos/sin: RoPE embeddings (Needed for the MLA RoPE head)
        kv_cache: The specific cache for this layer (Stores the compressed latents)
        start_pos: Current token position (for inference)
        """
        
        # --- STEP 1: ATTENTION RESIDUAL BLOCK ---
        # Formula: h = x + MLA(Norm(x))
        
        # A. Normalize
        normed_x = self.attention_norm(x)
        
        # B. Run MLA
        # We pass the cache and start_pos so MLA knows where to store/read the latent vector.
        attn_out = self.attention(
            normed_x, 
            freqs_cos, 
            freqs_sin, 
            kv_cache, 
            start_pos
        )
        
        # C. Residual Connection (The "+")
        # We add the new context info (attn_out) to the original info (x).
        h = x + attn_out

        # --- STEP 2: FEEDFORWARD RESIDUAL BLOCK ---
        # Formula: out = h + MoE(Norm(h))
        
        # A. Normalize
        normed_h = self.ffn_norm(h)
        
        # B. Run MoE
        # The Router inside SparseMoE picks the top-6 experts for these tokens.
        ffn_out = self.feed_forward(normed_h)
        
        # C. Residual Connection
        out = h + ffn_out

        return out
