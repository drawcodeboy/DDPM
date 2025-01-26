import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['RMSNorm']

class RMSNorm(nn.Module):
    # RMS Normalization is more simple than Group Normalization
    # However, that reasoning does not justify replacing Group Norm with RMS Norm. 
    # The EDM2 paper revealed that Group Norm did not perform well.
    # https://github.com/lucidrains/denoising-diffusion-pytorch/issues/11#issuecomment-2613964926
    # Thanks for reply, @lucidrains
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5 # Root
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # Gain Parameter

    def forward(self, x):
        # F.normalize's default p is 2.0. so, F.normalize mean L2 Norm
        # The intention was to normalize by the RMS using F.normalize, 
        # but it inadvertently divides by sqrt(n), where n is the vector dimension. 
        # To restore the non-RMS part, self.scale (sqrt(n)) is multiplied to recover the correct scale.
        return F.normalize(x, dim = 1) * self.g * self.scale