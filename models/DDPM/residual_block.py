import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple

from einops import rearrange

from .rms_norm import RMSNorm

__all__ = ['ResidualBlock']

class Block(nn.Module):
    def __init__(self, 
                 dim_in, 
                 dim_out, 
                 dropout_rate:float = 0): # in CIFAR 10, set 0.1
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU() # No Reason? just use?
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, scale_shift:Tuple = None):
        x = self.norm(self.conv(x))
        
        if scale_shift is not None:
            # Why is the Time Embedding not simply added, but instead chunked, scaled, and shifted before being applied?
            # https://github.com/lucidrains/denoising-diffusion-pytorch/issues/77
            # According to the issue above, simply adding provides only a linear effects, 
            # but combining it as shown below allows for non-linear effects, 
            # increasing the expressiveness that can be learned. 
            # Although there is no paper specifically addressing this, 
            # many studies have applied this approach.
            scale, shift = scale_shift # Tuple Unpacking
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return self.dropout(x)

class ResidualBlock(nn.Module):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 time_emb_dim:Optional[int]= None,
                 dropout_rate:float = 0.):
        super().__init__()
        
        self.time_emb_mlp = nn.Sequential(nn.SiLU(),
                                          nn.Linear(time_emb_dim, dim_out * 2), # for scale and shift
                                          ) if time_emb_dim is not None else None

        self.block1 = Block(dim_in=dim_in, 
                            dim_out=dim_out, 
                            dropout_rate=dropout_rate)
        
        self.block2 = Block(dim_in=dim_out,
                            dim_out=dim_out,
                            dropout_rate=dropout_rate)
        
        # for Residual connection
        self.residual_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x, time_emb = None):
        r = x.clone()
        
        scale_shift = None
        if (self.time_emb_mlp is not None) and (time_emb is not None):
            scale_shift = self.time_emb_mlp(time_emb)
            scale_shift = rearrange(scale_shift, 'b c -> b c 1 1')
            scale_shift = scale_shift.chunk(2, dim=1)
        
        x = self.block1(x, scale_shift=scale_shift)
        x = self.block2(x)
        
        return x + self.residual_conv(r)