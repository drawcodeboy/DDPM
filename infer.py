import torch
import torch.distributions as dist

import cv2
from einops import rearrange

import yaml
import argparse

from datasets import load_dataset
from models import load_model
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--config', type=str, default='base')
    return parser

def main(cfg, args):
    print(f"=================[{cfg['expr']}]=================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else:
        device = 'cpu'
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(**model_cfg).to(device)
    ckpt = torch.load(os.path.join(cfg['save_path'], cfg['load_weights']),
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Load Model {model_cfg['name']}")
    
    timesteps = model.algorithm2(shape=[1, 1, 28, 28],
                                 get_all_timesteps=True,
                                 verbose=True)
    to_npy = lambda x: (rearrange(x.cpu().detach().numpy(), '1 c h w -> h w c') * 255.).astype(np.uint8)
    timesteps = list(map(to_npy, timesteps))
    
    if not os.path.exists('./generated_samples'):
        os.mkdir('./generated_samples')
    
    for idx, x in enumerate(timesteps, start=1):
        cv2.imwrite(f'./generated_samples/sample_{idx:04d}.jpg', x)
    
    idx = 0
    while True:
        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('images', width=600, height=600)
        cv2.imshow('images', timesteps[idx])
        if cv2.waitKeyEx(1) == 27:
            break
        
        idx += 1
        if idx >= len(timesteps):
            idx = 0
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    
    with open(f'config/infer.{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg, args)