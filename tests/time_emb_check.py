import os, sys
sys.path.append(os.getcwd())

import torch
from models.DDPM.time_embedding import SinusoidalPosEmbedding

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