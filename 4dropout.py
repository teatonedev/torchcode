import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p ==0.0:
            return x

        mask = (torch.rand_like(x) >= self.p).to(dtype=x.dtype)

        return ( torch.mul(x,mask) ) / (1.0 - self.p)




d = MyDropout(p=0.5)
d.train()
x = torch.ones(100)

# now lets do a large train :D
print('Train: ', d(x))
d.eval()
print('Eval: ', d(x))

# at the end train values must be either dropped to 0 or
# to scaled with 1 / (1 - p) 
