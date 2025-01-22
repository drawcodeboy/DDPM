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
    
if __name__ == '__main__':
    temp = 20
    t = t = torch.arange(temp).long()
    pos_emb = SinusoidalPosEmbedding(dim=256)
    emb = pos_emb(t)
    
    print(emb.shape)
    for i in range(0, temp-1):
        # Similarity using Manhattan Distance
        dist_1 = torch.norm((emb[i] - emb[i+1]), p = 1) 
        dist_1 = 1/(1+dist_1)
        
        # Similarity using Euclidean Distance
        dist_2 = torch.norm((emb[i] - emb[i+1]), p = 2) 
        dist_2 = 1/(1+dist_2)
        
        # Cosine Similarity
        cos_s = torch.dot(emb[i], emb[i+1]) / (torch.norm(emb[i], p=2) * torch.norm(emb[i+1], p=2))
        print(f"Similarity: {dist_1.item():.2f}, {dist_2.item():.2f}, {cos_s.item():.2f}")
        
        # Manhattan Distance
        dist_m = torch.abs(emb[i] - emb[i+1]).sum()
        
        # Euclidean Distance
        dist_e = torch.pow(emb[i] - emb[i+1], 2).sum()
        print(f"Distance: {dist_m.item():.2f}, {dist_e.item():.2f}")