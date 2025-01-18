import torch
from torch import nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self,
                 time_steps: int = 1000,
                 beta_schedule: str = 'linear'):
        super().__init__()
        
        # 1) Alpha & Beta Settings
        if beta_schedule == 'linear':
            betas = self._beta_linear_schedule(time_steps)
        else:
            raise Exception(f"beta_schdule {beta_schedule} is not supported yet.")
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # Cumulative Product
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # 2) Coefficients Using Alpha & Beta
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.-alphas_cumprod))
        
        # 3) Time Step Settings
        self.time_steps = time_steps
        
    def _beta_linear_schedule(self, time_steps):
        start, end = 0.0001, 0.02
        
        return torch.linspace(start, end, time_steps)
    
    def _algorithm_1(self, x_0):
        # [1] Repeat
        # [2] x_0 ~ q(x_0)
        
        # [3] t ~ Uniform({1, ..., T})
        # Sampling t from Uniform dist, without 0
        # Because x_0 prediction needs discerete decoder.
        bz = x_0.shape[0] # Batch Size
        t = torch.randint(1, self.time_steps, (bz,), device=x.device)
        
        # [4] Noise(Epsilon) ~ N(0,I)
        noise = torch.randn_like(x_0) # I = Identity matrix(Covariance Matrix)
        
        # [5] Get Loss
        # [5.1] Get Noise_{theta}; x_t=sqrt{bar{alpha_t}x_0+sqrt{1-bar{alpha_t}}epsilon
    
    def forward(self, x):
        self._algorithm_1(x_0)

        return t

if __name__ == '__main__':
    model = DDPM()
    
    time_steps = model(torch.randn(3, 3))
    print(time_steps)