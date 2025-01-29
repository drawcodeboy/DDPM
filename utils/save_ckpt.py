import os
import torch
import numpy as np
import re

def save_model_ckpt(model_name,
                    dataset,
                    current_epoch,
                    steps_per_epoch,
                    model,
                    save_path,
                    leave_best=False):
    
    model_name, dataset = model_name.lower(), dataset.lower()
    
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['steps'] = steps_per_epoch * current_epoch
    
    # eliminate previous epoch
    if leave_best==True:
        prev_pths = os.listdir(save_path)
        
        for prev_pth in prev_pths:
            prev_parsed = prev_pth.split('.')
            if prev_parsed[-1] != "pth": # pass, if file extension is not pth
                continue
            
            prev_model = prev_parsed[0]
            prev_dataset = prev_parsed[1]
            prev_epoch = int("".join(re.findall(r'\d+', prev_parsed[2])))
            
            if prev_model == model_name and prev_dataset == dataset:
                if prev_epoch < current_epoch:
                    # eliminate if this file low epoch than current epoch
                    os.remove(os.path.join(save_path, prev_pth))
    
    state_dict_name = f'{model_name}.{dataset}.{steps_per_epoch * current_epoch:07d}steps.pth'
    
    try:
        torch.save(ckpt, os.path.join(save_path, state_dict_name))
        print(f"Save Model @epoch: {current_epoch}")
    except:
        print(f"Can\'t Save Model @epoch: {current_epoch}")


def save_loss_ckpt(model_name,
                   dataset,
                   train_loss,
                   save_path):
    
    model_name, dataset = model_name.lower(), dataset.lower()
    
    try:
        np.save(os.path.join(save_path, f'train_loss.{model_name}.{dataset}.npy'), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss')