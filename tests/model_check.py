import torch
from torch import nn

import yaml

import os, sys
sys.path.append(os.getcwd())

from models import load_model

if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.full_load(f)
    model_cfg = cfg['model']
    model = load_model(**model_cfg)
    
    print(model)