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
        self.qkv = nn.Conv2d(dim_in, 3*attn_emb_dim, 1)
        self.attn = nn.MultiheadAttention(embed_dim=attn_emb_dim,
                                          num_heads=attn_heads,
                                          dropout=dropout_rate,
                                          batch_first=True)
        self.to_out = nn.Conv2d(attn_emb_dim, dim_in, 1) # Reconstrct dimension size: Attention Emb -> Input Emb
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        
        x = self.qkv(x) # Make Query, Key, Value Embedding
        x = rearrange(x, 'b c h w -> b (h w) c') # Flatten
        q, k, v = torch.chunk(x, 3, dim=-1) # Chunk query, Key, Value Embedding
        
        x, _ = self.attn(q, k, v)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.to_out(x)
        return x