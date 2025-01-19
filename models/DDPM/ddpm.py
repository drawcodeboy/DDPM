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
        
        return torch.linspace(start, end, time_steps) # [start, ..., end], includes end
    
    def _algorithm_1(self, x_0):
        # [1] Repeat
        # [2] x_0 ~ q(x_0)
        
        # [3] t ~ Uniform({1, ..., T})
        # Sampling t from Uniform dist, without 0
        # Because x_0 prediction needs discerete decoder.
        
        # REMEMBER!
        # The range of torch.randint() is [0, 999] instead of [1, 1000] because, 
        # during value extraction using gather, AN INDEX OF 0 CORRESPONDS TO A TIME STEP OF 1. 
        # This is a common trick, but due to the complexity of the implementation, 
        # it may cause confusion, so this comment is added for clarity.
        bz = x_0.shape[0] # Batch Size
        t = torch.randint(0, self.time_steps, (bz,), device=x_0.device, dtype=torch.int64) # [0, self.time_steps - 1]
        
        # [4] Noise(Epsilon) ~ N(0,I)
        noise = torch.randn_like(x_0) # I = Identity matrix(Covariance Matrix)
        
        # [5] Get Loss
        # [5.1] model(U-Net) input=x_t=sqrt_alphas_cumprod*x_0+sqrt_one_minus_alphas_cumprod*noise
        x_0_coef = self.sqrt_alphas_cumprod.gather(-1, t) # Extract value corresponding to time step
        x_0_coef = x_0_coef.reshape(bz, *((1,) * (len(x_0.shape)-1)))
        
        noise_coef = self.sqrt_one_minus_alphas_cumprod.gather(-1, t)
        noise_coef = noise_coef.reshape(bz, *((1,) * (len(x_0.shape)-1)))
        
        x_t = x_0_coef * x_0 + noise_coef * noise
        # [5.2] model(U-Net) output
        print(t.shape, x_t.shape)
    
    def forward(self, x_0):
        self._algorithm_1(x_0)

        return x_0

if __name__ == '__main__':
    model = DDPM()
    
    time_steps = model(torch.randn(4, 3, 16, 16))