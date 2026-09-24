#Creating tensors

import torch

x=torch.tensor(5)
y=torch.tensor([1,23,44])
z=torch.tensor([[1,2],[3,4]])

print(x,y,z)
print(x.shape,y.shape,z.shape)