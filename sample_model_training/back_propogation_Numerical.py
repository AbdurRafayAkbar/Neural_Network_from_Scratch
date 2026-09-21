#inputs
x=2
w=3
b=1
lr=0.01
#target
y=10
#Forward pass
predicted=(x*w)+b

#Activation function (ReLU)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from activationfunctions import Relu
from activationfunctions import Relu_derivative
a=Relu(predicted)

#Loss
Loss=(predicted-y)**2

#Backward pass
dloss_dpredicted=2*(predicted-y)

#Relu_derivative
drelu_dpredicted=Relu_derivative(a)

#Weight Gradient

dw=dloss_dpredicted*drelu_dpredicted*x

#Bias Gradient

db=dloss_dpredicted*drelu_dpredicted*b

#gradient descent

w=w-(lr*dw)
b=b-(lr*db)

#output
print(f"Updated weight: {w}")
print(f"Updated bias: {b}")
print(f"Updated predicted: {predicted}")
