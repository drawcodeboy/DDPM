import torch
from torch import nn

from einops.layers.torch import Rearrange

class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, contract=True):
        super().__init__()
        
        self.down = None
        if contract == True:
            self.down = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b c h w', p1=2, p2=2),
                                      nn.Conv2d(dim_in, dim_out, 3, padding=1))
        else:
            self.down = nn.Conv2d(dim_in, dim_out, 3, padding=1)
    
    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, expand=True):
        super().__init__()
        
        self.up = None
        if expand == True:
            self.up = nn.ConvTranspose2d(dim_in, dim_out, 2, stride=2)
        else:
            self.up = nn.Conv2d(dim_in, dim_out, 3, padding=1)
    
    def forward(self, x):
        return self.up(x)