import torch
from torch import nn

import yaml

import os, sys
sys.path.append(os.getcwd())

from models import load_model

if __name__ == '__main__':
    with open('config/train.small.yaml') as f:
        cfg = yaml.full_load(f)
    model_cfg = cfg['model']
    model = load_model(**model_cfg)
    
    print(model)
    
    loss = model.algorithm1(torch.randn(1, 1, 16, 16))
    
    total_sum = 0
    for params in model.parameters():
        total_sum += params.numel()
    print(f"Model Parameters: {total_sum}")