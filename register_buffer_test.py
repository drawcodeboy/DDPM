import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tensor = torch.randn(3, 3)
        
        self.param = nn.Parameter(torch.randn(3, 3))
        
        buff = torch.randn(3, 3)
        self.register_buffer('buff', buff)
    
    def forward(self, x):
        print(self.buff)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)

print("===========================================")
print(f"tensor: \n{model.tensor}")
print(f"{model.tensor.device}")

print("===========================================")
for name, parameter in model.named_parameters():
    print(f"{name}: \n{parameter}")
    print(f"{parameter.device}")

print("===========================================")
for name, buff in model.named_buffers():
    print(f"{name}: \n{buff}")
    print(f"{buff.device}")

print("===========================================")
print(f"model state_dict(): param O, buff O, tensor X")
print(model.state_dict())

from torch.distributions.normal import Normal

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gaussian = Normal(loc=torch.zeros(3),
                               scale=torch.ones(3))
        
        gaussian2 = Normal(loc=torch.zeros(3),
                           scale=torch.ones(3))
        
        self.register_buffer('gaussian2', gaussian2)
    
    def forward(self, x):
        sample = self.gaussian.sample()
        
        sample2 = self.gaussian2.sample()
        
        print(f"sample 1 device: {sample.device}")
        print(f"sample 2 device: {sample2.device}")
        
        return x

model2 = Model2()
model2(torch.randn(3, 3))
# Distribution은 buffer에 register 안 된다.