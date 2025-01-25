# Why Scale & Shift


# Attention is appled for all layers

# Why RMS, RMS is simply

import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(Module):
    # RMS Normalization is more simple than Group Normalization
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
    
    def forward(self, x):
        x = self.norm(self.conv(x))
        
        if scale_shift is not None:
            # Why is the Time Embedding not simply added, but instead chunked, scaled, and shifted before being applied?
            # https://github.com/lucidrains/denoising-diffusion-pytorch/issues/77
            # According to the issue above, simply adding provides only a linear effects, 
            # but combining it as shown below allows for non-linear effects, 
            # increasing the expressiveness that can be learned. 
            # Although there is no paper specifically addressing this, 
            # many studies have applied this approach.
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return self.dropout(x)