# Expr 1
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

# Expr 2
# Distribution은 buffer에 register 안 된다.
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

'''
===========================================
tensor: 
tensor([[ 0.1708, -0.5264, -1.1837],
        [ 0.4366,  0.1491,  1.5451],
        [ 1.6670, -0.2930,  0.3820]])
cpu
===========================================
param:
Parameter containing:
tensor([[ 0.3108,  0.5101, -0.8290],
        [ 0.3511, -0.4658,  0.2131],
        [ 0.6602, -1.0786, -0.3299]], requires_grad=True)
cpu
===========================================
buff:
tensor([[ 0.3143,  0.1681,  0.0056],
        [ 0.3792,  1.1674,  1.1615],
        [ 0.6245,  1.2716, -0.8061]])
cpu
===========================================
OrderedDict({'param': tensor([[ 0.3108,  0.5101, -0.8290],
        [ 0.3511, -0.4658,  0.2131],
        [ 0.6602, -1.0786, -0.3299]]), 'buff': tensor([[ 0.3143,  0.1681,  0.0056],
        [ 0.3792,  1.1674,  1.1615],
        [ 0.6245,  1.2716, -0.8061]])})

(.venv) E:\DDPM>python register_buffer_test.py
===========================================
tensor: 
tensor([[ 0.5325,  1.3698, -1.2790],
        [-0.5546,  0.3236,  0.6196],
        [ 2.1521,  1.8287,  1.0600]])
cpu
===========================================
param:
Parameter containing:
tensor([[-1.5823, -1.0639, -0.9007],
        [-0.8665, -0.0151,  0.8802],
        [ 0.3128,  2.1903, -0.2867]], requires_grad=True)
cpu
===========================================
buff:
tensor([[-0.0590,  0.4823,  2.1716],
        [-0.6110,  0.1420,  1.5730],
        [ 1.2040, -0.0654, -0.4525]])
cpu
===========================================
model state_dict(): param O, buff O, tensor X
OrderedDict({'param': tensor([[-1.5823, -1.0639, -0.9007],
        [-0.8665, -0.0151,  0.8802],
        [ 0.3128,  2.1903, -0.2867]]), 'buff': tensor([[-0.0590,  0.4823,  2.1716],
        [-0.6110,  0.1420,  1.5730],
        [ 1.2040, -0.0654, -0.4525]])})
Traceback (most recent call last):
  File "E:\DDPM\register_buffer_test.py", line 64, in <module>
    model2 = Model2()
             ^^^^^^^^
  File "E:\DDPM\register_buffer_test.py", line 52, in __init__
    self.register_buffer('gaussian2', gaussian2)
  File "E:\DDPM\.venv\Lib\site-packages\torch\nn\modules\module.py", line 566, in register_buffer
    raise TypeError(
TypeError: cannot assign 'torch.distributions.normal.Normal' object to buffer 'gaussian2' (torch Tensor or None required)
'''