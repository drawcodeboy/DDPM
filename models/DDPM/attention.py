import torch
from torch import nn

from einops import rearrange

from .rms_norm import RMSNorm

__all__ = ['Attention']

class Attention(nn.Module):
    def __init__(self,
                 dim_in:int,
                 attn_emb_dim:int,
                 attn_heads:int,
                 dropout_rate:float):
        super().__init__()
        
        self.norm = RMSNorm(dim=dim_in)
        self.attn = nn.MultiheadAttention(embed_dim=attn_emb_dim,
                                          num_heads=attn_heads,
                                          dropout=dropout_rate,
                                          batch_first=True)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.norm(x)
        
        x = rearrange(x, 'b c h w -> b (h w) c') # Flatten
        
        x = self.attn(x)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        
        return x