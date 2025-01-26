# Attention is appled for all layers
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple

__all__ = ['ResidualBlock']

class RMSNorm(nn.Module):
    # RMS Normalization is more simple than Group Normalization
    # However, that reasoning does not justify replacing Group Norm with RMS Norm. 
    # The EDM2 paper revealed that Group Norm did not perform well.
    # https://github.com/lucidrains/denoising-diffusion-pytorch/issues/11#issuecomment-2613964926
    # Thanks for reply, @lucidrains
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5 # Root
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # Gain Parameter

    def forward(self, x):
        # F.normalize's default p is 2.0. so, F.normalize mean L2 Norm
        # The intention was to normalize by the RMS using F.normalize, 
        # but it inadvertently divides by sqrt(n), where n is the vector dimension. 
        # To restore the non-RMS part, self.scale (sqrt(n)) is multiplied to recover the correct scale.
        return F.normalize(x, dim = 1) * self.g * self.scale

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
        
        # Time Embedding MLP: Personal Opinion
        # Time Embedding changes linearly over time, but this linearity alone is insufficient 
        # to convey how much noise should be removed at each time step. 
        # Therefore, a nonlinear transformation is needed. 
        # By applying SiLU before the linear transformation, 
        # the model can learn complex patterns and effectively predict the amount of noise to remove.
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
        res = x.clone()
        
        scale_shift = None
        if (self.time_emb_mlp is not None) and (time_emb is not None):
            scale_shift = self.time_emb_mlp(time_emb)
            scale_shift = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = scale_shift.chunk(2, dim=1)
        
        x = self.block1(x, scale_shift=scale_shift)
        x = self.block2(x)
        
        return x + self.residual_conv(res)