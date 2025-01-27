import sys, os
sys.path.append(os.getcwd())

import torch

import cv2
import numpy as np
from einops import rearrange
import yaml
import argparse

from models import load_model
from datasets import load_dataset

if __name__ == '__main__':
    with open('config/train.small.yaml') as f:
        cfg = yaml.full_load(f)
    model_cfg = cfg['model']

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()

    model = load_model(**model_cfg)

    ds = load_dataset()
    x_0, label = ds[0]
    x_0 = x_0.unsqueeze(0)

    x_t = model.view(x_0)
    x_t = torch.chunk(x_t, 1000)

    x_t = x_t[::-1] # Reverse

    if not os.path.exists('./samples'):
        os.mkdir('./samples')

    print('Complete All Settings')

    for idx, x in enumerate(x_t, start=1):
        x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().cpu().numpy()
        x = (x * 255.).astype(np.uint8)
        cv2.imwrite(f'./samples/sample_{idx:04d}.jpg', x)

    img_path_li = [os.path.join('./samples', path) for path in os.listdir('./samples')]

    img_path_li = img_path_li[::args.steps]

    idx = 0
    while True:
        img = cv2.imread(img_path_li[idx])
        
        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('images', width=600, height=600)
        cv2.imshow('images', img)
        if cv2.waitKeyEx(1) == 27:
            break
        
        idx += 1
        if idx >= len(img_path_li):
            idx = 0

    cv2.destroyAllWindows()