import torch
from torch import nn
import math

class SinusoidalPosEmbedding(nn.Module):
    # Time Embedding
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
    
    def forward(self, t):
        '''
            t: (bz,) - time steps sampling from Uniform distribution. 
        '''
        half_dim = self.dim // 2 # 2i/d -> i/(d/2)
        # Why divide (half_dim - 1), not half_dim?????
        # Because, dimension starts from 0 to half_dim - 1.
        # If use d/2 = half_dim, exponent can't reach 1!
        # i.e. It cannot achieve the intended frequency range. [0, 1/10000]
        # Unique values and similar intervals between positions are guaranteed, 
        # so dividing by half_dim - 1 is not strictly necessary. 
        # However, it is done to explicitly implement the intended design.
        
        emb = math.log(self.theta) / (half_dim - 1) # log(10000^(2i/d))
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb) # e^(-log(10000^(2i/d))) -> 1/(10000^(2i/d))
        emb = t[:, None] * emb[None, :] # (t,) * (emb,) -> (t, 1) * (1, emb) -> (t, emb) | time step assigment
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # 1/(10000^(2i/d)) -> sin(1/(10000^(2i/d))), cos(1/(10000^(2i/d)))
        return emb