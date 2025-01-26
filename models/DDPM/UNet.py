import torch
from torch import nn

from typing import Tuple

from .residual_block import ResidualBlock

class UNet(nn.Module):
    def __init__(self,
                 input_dim:int, # Input Data Dimension
                 init_dim:int, # Start with this dimension, it effects on overall model dimension
                 dim_mults:Tuple = (1, 2, 4, 8),
                 time_emb_dim:int = 16,
                 dropout_rate:float = 0., # In CIFAR 10, 0.2
                 ):
        super().__init__()
        
        # [1] Determine Comprehensive Dimensions
        
        # Start with this dimension init_dim
        self.init_conv = nn.Conv2d(input_dim, init_dim, 7, padding = 3)
        
        # In U-Net, the depth of the feature map doubles as it moves through the contracting path 
        # starting from the initial dimension(init_dim), and this operation is performed for this reason.
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        
        in_out = list(zip(dims[:-1], dims[1:])) # pair(idx: 0~3, idx: 1:4)
        
        # [2] Layers
        # [2.1] Contracting Path
        self.downs = nn.ModuleList()
        
        for (dim_in, dim_out) in in_out:
            self.downs.append(nn.ModuleList(
                ResidualBlock(dim_in=dim_in, 
                              dim_out=dim_in,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                ResidualBlock(dim_in=dim_in, 
                              dim_out=dim_in,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                
            ))
        
    
    def forward(self, x, t):
        return x
    