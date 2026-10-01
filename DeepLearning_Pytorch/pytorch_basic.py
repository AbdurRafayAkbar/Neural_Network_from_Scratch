#Creating tensors

import torch
from torch import dtype 
x=torch.tensor(5)
y=torch.tensor([1,23,44])
z=torch.tensor([[1,2],[3,4]])

# print(x,y,z)
# print(x.shape,y.shape,z.shape)

#------------
#D_type of tensor
a=torch.tensor([2,3,4])
print(a.dtype) #int64
b=torch.tensor([2.0,3.0,4.0])   
print(b.dtype)