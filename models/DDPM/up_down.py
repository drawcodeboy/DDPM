import torch
from torch import nn

from einops.layers.torch import Rearrange

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.down = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b c h w', p1=2, p2=2),
                                  nn.Conv2d(dim, dim, 3, padding=1))
    
    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(dim, dim, 2, stride=2)
    
    def forward(self, x):
        return self.up(x)