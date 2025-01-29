import torch
from torch.utils.data import dataset
from torchvision.datasets import MNIST
import numpy as np

class MNIST_Dataset():
    def __init__(self,
                 root="data/", # make MNIST directory below data/
                 download=True, # download if there is no data, else pass
                 mode='train'): 
        
        if mode not in ['train', 'test']:
            raise AssertionError(f"mode 확인: [{mode}]으로 입력 됨")
        
        self.data = MNIST(root=root,
                    download=download)
        
        self.data=list(self.data)
        
        train_size = 5000
        test_size = 500
        
        if mode == 'train':
            self.data = self.data[:train_size]
        elif mode == 'test':
            self.data = self.data[train_size:train_size+test_size]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        image = np.array(image.getdata()).reshape(28, 28).astype(np.float32)
        image /= 255.
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label