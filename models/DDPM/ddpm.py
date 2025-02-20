import torch
from torch import nn
import torch.nn.functional as F

from einops import reduce

from typing import Tuple
from functools import partial

from .unet import UNet

class DDPM(nn.Module):
    def __init__(self,
                 time_steps: int = 1000, # DDPM
                 beta_schedule: str = 'linear', # DDPM
                 input_dim:int = 1, # U-Net
                 init_dim:int = 1, # U-Net
                 dim_mults:Tuple = (1, 2, 4, 8), # U-Net
                 time_emb_theta:int = 10000, # U-Net
                 time_emb_dim:int = 16, # U-Net
                 attn_emb_dim:int = 32, # U-Net
                 attn_heads:int = 4, # U-Net
                 dropout_rate:float = 0., # U-Net
                 device: str = 'cuda'): # Overall
        super().__init__()
        
        # 1) Alpha & Beta Settings
        if beta_schedule == 'linear':
            betas = self._beta_linear_schedule(time_steps)
        else:
            raise Exception(f"beta_schdule {beta_schedule} is not supported yet.")
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # Cumulative Product
        
        self.register_buffer('alphas', alphas.to(torch.float32))
        self.register_buffer('betas', betas.to(torch.float32))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(torch.float32))
        
        # 2.1) Coefficients Using Alpha & Beta
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32)) # Used in Algorithm 1
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.-alphas_cumprod).to(torch.float32)) # Used in Algorithm 1, 2
        
        # 2.2) P Variance (used in Algorithm 2)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        p_var = betas * (1.-alphas_cumprod_prev) / (1.-alphas_cumprod)
        self.register_buffer('p_var', p_var.to(torch.float32))
        # In log calculation, 0 will go to -inf, and variance can get 0. In brief, we need clamp
        self.register_buffer('p_log_var_clipped', torch.log(p_var.clamp(min=1e-20)).to(torch.float32))
        
        # 3) Time Step Settings
        self.time_steps = time_steps
        
        # 4) Get Model (U-Net)
        self.model = UNet(input_dim=input_dim,
                          init_dim=init_dim,
                          dim_mults=dim_mults,
                          time_emb_theta=time_emb_theta,
                          time_emb_dim=time_emb_dim,
                          attn_emb_dim=attn_emb_dim,
                          attn_heads=attn_heads,
                          dropout_rate=dropout_rate)
        
        # Extra
        self.device = device
        
    def _beta_linear_schedule(self, time_steps):
        start, end = 0.0001, 0.02
        
        return torch.linspace(start, end, time_steps, dtype=torch.float64) # [start, ..., end], includes end
    
    def _normalize(self, x):
        # DDPM needs [-1, 1] scale input, it is related to All process performs N(0,I)
        # Mostly, Image data is normalized [0, 1]. So, I re-scaled [0, 1] to [-1, 1]
        return (x * 2) - 1
    
    def _unnormalize(self, x):
        # reconstruct scale [-1, 1] to [0, 1]
        return (x + 1) * 0.5
    
    def _extract(self, a, t, x_shape):
        '''
            t.shape = (b,)
            a.shape = (self.num_timesteps,)
            out.shape = (b,)
            return out.shape = (b, 1, 1, ..., 1)
        '''
        b, *_ = t.shape
        out = a.gather(dim=-1, index=t) # choose value a from index t -> out = (b,)
        return out.reshape(b, *((1,) * (len(x_shape)-1)))
    
    def algorithm1(self, x_0):
        # Training
        
        # [1] Repeat
        
        # [2] x_0 ~ q(x_0)
        x_0 = self._normalize(x_0) # scale: [0, 1] -> [-1, 1], this code is not related to this part[2], just pre-processing.
        
        # [3] t ~ Uniform({1, ..., T})
        # Sampling t from Uniform dist, without 0
        # Because x_0 prediction needs discerete decoder.
        
        # REMEMBER!
        # The range of torch.randint() is [0, 999] instead of [1, 1000] because, 
        # during value extraction using gather, AN INDEX OF 0 CORRESPONDS TO A TIME STEP OF 1. 
        # This is a common trick, but due to the complexity of the implementation, 
        # it may cause confusion, so this comment is added for clarity.
        # When creating Time Embeddings, it's still fine to input 0 for time step 1 
        # because the purpose of Time Embedding is to maintain relative differences, 
        # i.e., the "INTERVAL AND SCALE EQUIVALENCE," rather than 
        # requiring exact corresponding values for each step.
        
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
        noise_pred = self.model(x_t, t)
        # [5.3] loss
        loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean') # >> consider loss_weight... so I get mean twice.
        
        return loss.mean()
    
    def p_mean_variance(self, x_t, t):
        # [4] get x_{t-1}
        noise_pred = self.model(x_t, t)

        p_mean = x_t - (self._extract(self.betas, t, x_t.shape) / self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)) * noise_pred
        p_mean = p_mean / self._extract(self.alphas, t, x_t.shape)
        
        # Variance is fixed value
        p_var = self._extract(self.p_var, t, x_t.shape)
        p_log_var_clipped = self._extract(self.p_log_var_clipped, t, x_t.shape)
        
        return p_mean, p_var, p_log_var_clipped
    
    @torch.inference_mode()
    def sample(self, x_t, t, clip_scaled=True):
        # [3] z ~ N(0, I)
        z = torch.randn_like(x_t)

        # [4] get x_{t-1}
        t_ = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.int64)
        
        # this is KEY POINT!
        p_mean, p_var, p_log_var = self.p_mean_variance(x_t=x_t, t=t_)

        x_t_minus_one = p_mean + ((0.5 * p_log_var).exp() * z if t != 0 else 0)
        
        if clip_scaled:
            # It guarantees sampling quality.
            # Because, it trained data scale [-1, 1]
            x_t_minus_one.clamp_(-1., 1.)
        
        return x_t_minus_one
    
    @torch.inference_mode()
    def algorithm2(self, shape, get_all_timesteps=False, verbose=False):
        '''
            Usage: model.algorithm2()
            shape: output.shape you want. if shape is [4, 3, 128, 128], 4 images that [3, 128, 128] shape.
        '''
        # Sampling
        
        # [1] x_T ~ N(0, I)
        x_t = torch.randn(shape, device = self.device)
        
        all_timesteps = [torch.clamp(x_t, -1., 1.)] # for visualize, torch.clamp()
        
        # [2] for t = T,...,1 do 
        for t in range(self.time_steps-1, 0, -1): # [self.timesteps-1, 1], time step 1 for discrete decoder.
            # [3], [4] sampling z, and get x_{t-1}
            x_t = self.sample(x_t, t)
            
            all_timesteps.append(x_t)
            if verbose == True:
                print(f"\rGet Time Steps: [{t:04d}] -> [{t-1:04d}]", end="") # Actually, Predict from t to t-1
        if verbose == True: print()
        
        # [5] end for, [6] return x_0 (x_1 for discrete decoder)
        if not get_all_timesteps:
            x_1 = self.unnormalize(torch.clamp(x_t, -1., 1.))
            return x_1
        else:
            clip = partial(torch.clamp, min=-1, max=1.)
            all_timesteps = list(map(clip, all_timesteps))
            all_timesteps = list(map(self._unnormalize, all_timesteps))
            return all_timesteps
        
    def forward(self, x_0):
        loss = self.algorithm1(x_0)

        return loss
        
    def view(self, x_0):
        # view forward or reverse process (that depends on post-processing)
        # Not used in training or sampling
        x_0 = self._normalize(x_0)
        noise = torch.randn_like(x_0)
        
        x_0_coef = self.sqrt_alphas_cumprod.reshape(-1, 1, 1, 1)
        noise_coef = self.sqrt_one_minus_alphas_cumprod.reshape(-1, 1, 1, 1)
        
        x_t = x_0_coef * x_0 + noise_coef * noise
        x_t = torch.clamp(x_t, min=-1., max=1.) # for visualize, min-max clamp
        
        x_t = self._unnormalize(x_t)
        return x_t