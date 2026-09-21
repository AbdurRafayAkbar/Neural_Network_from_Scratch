#To get importing files
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

#inputs
from activationfunctions import Relu, Relu_derivative


x1=2
x2=4
x3=6

#weights

w1=3
w2=5
w3=7

#Bias
b=5

#Learning rate
lr=0.01

#Target

y=20

#Forward pass

z=(x1*w1)+(x2*w2)+(x3*w3)+b

#Relu activation function

a=Relu(z)

# print(f"Predicted value: {a}")
# print(f"forward pass value: {z}")
# print(f"Target value: {y}")

#Measuring loss

loss=(a-y)**2

#Measring Backpropogation direction

dl_da=2*(a-y)

print(loss)
print(dl_da)