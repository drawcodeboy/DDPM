import torch
from torch import nn

from typing import Tuple

from .residual_block import ResidualBlock
from .attention import Attention
from .up_down import DownSample, UpSample
from .time_embedding import SinusoidalPosEmbedding

class UNet(nn.Module):
    def __init__(self,
                 input_dim:int, # Input Data Dimension
                 init_dim:int = 1, # Start with this dimension, it effects on overall model dimension
                 dim_mults:Tuple = (1, 2, 4, 8),
                 time_emb_dim:int = 16, # Not origin dimension, this dimension is after pass through Time MLP 
                 time_emb_theta:int = 10000,
                 attn_emb_dim:int = 32,
                 attn_heads:int = 4,
                 dropout_rate:float = 0., # In CIFAR 10, 0.2
                 ):
        super().__init__()
        
        # [1] Determine Comprehensive Dimensions
        # [1.1] Start with this dimension init_dim
        self.init_conv = nn.Conv2d(input_dim, init_dim, 7, padding = 3)
        
        # [1.2] U-Net Dimension per stage
        # In U-Net, the depth of the feature map doubles as it moves through the contracting path 
        # starting from the initial dimension(init_dim), and this operation is performed for this reason.
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        
        in_out = list(zip(dims[:-1], dims[1:])) # pair(idx: 0~3, idx: 1:4)
        
        # [1.3] Attention information
        attn_emb_dim_per_stage = (attn_emb_dim,) * len(dim_mults)
        attn_heads_per_stage = (attn_heads,) * len(dim_mults)
        attn_infos_per_stage = list(zip(attn_emb_dim_per_stage, attn_heads_per_stage))
        
        # [2] Layers
        # [2.1] Contracting Path
        self.downs = nn.ModuleList()
        
        for idx, ((dim_in, dim_out), (attn_emb_dim, attn_heads)) in enumerate(zip(in_out, attn_infos_per_stage)):
            is_last = (idx == len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in=dim_in, 
                              dim_out=dim_in,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                ResidualBlock(dim_in=dim_in, 
                              dim_out=dim_in,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                Attention(dim_in=dim_in,
                          attn_emb_dim=attn_emb_dim,
                          attn_heads=attn_heads,
                          dropout_rate=dropout_rate),
                DownSample(dim_in=dim_in, # Actually, Contracting is here.
                           dim_out=dim_out,
                           contract=(not is_last))
            ]))
        
        # [2.2] Mid
        mid_dim = dims[-1]
        self.mid = nn.ModuleList([ResidualBlock(mid_dim, mid_dim),
                                  Attention(dim_in=mid_dim,
                                            attn_emb_dim=attn_emb_dim_per_stage[-1],
                                            attn_heads=attn_heads_per_stage[-1],
                                            dropout_rate=dropout_rate),
                                  ResidualBlock(mid_dim, mid_dim)])
        
        # [2.3] Expanding Path
        self.ups = nn.ModuleList()
        
        for idx, ((dim_in, dim_out), (attn_emb_dim, attn_heads)) in enumerate(zip(*map(reversed, (in_out, attn_infos_per_stage)))):
            is_last = (idx == len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_in=dim_out+dim_in, # Previous layer + Skip
                              dim_out=dim_out,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                ResidualBlock(dim_in=dim_out+dim_in, # Previous layer + Skip
                              dim_out=dim_out,
                              time_emb_dim=time_emb_dim,
                              dropout_rate=dropout_rate),
                Attention(dim_in=dim_out,
                          attn_emb_dim=attn_emb_dim,
                          attn_heads=attn_heads,
                          dropout_rate=dropout_rate),
                UpSample(dim_in=dim_out, # Actually, Expanding is here
                         dim_out=dim_in,
                         expand=(not is_last))
            ]))
        
        # [2.4] Final
        self.final = nn.ModuleList([ResidualBlock(dim_in=init_dim * 2,
                                                  dim_out=init_dim,
                                                  dropout_rate=dropout_rate),
                                    nn.Conv2d(init_dim, input_dim, 1)])
        
        # [2.5] Time Embedding & Time MLP
        # Time Embedding is passed through an MLP and transformed into 
        # a dimensionality that is input into each Block. 
        # Therefore, the dimensionality of the original Time Embedding must be defined. 
        # Here, it is set to init_dim * 4.
        time_emb_origin_dim = init_dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmbedding(dim=time_emb_origin_dim,
                                                             theta=time_emb_theta),
                                      nn.Linear(time_emb_origin_dim, time_emb_dim),
                                      nn.GELU(),
                                      nn.Linear(time_emb_dim, time_emb_dim))
    
    def forward(self, x, t):
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(t)
        
        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            
            x = downsample(x)
        
        mid_block1, mid_attn, mid_block2 = self.mid
        x = mid_block1(x, t)
        x = mid_attn(x) + x
        x = mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        
        final_block, final_conv = self.final
        x = final_block(x, t)
        x = final_conv(x)
        
        return x