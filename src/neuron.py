#inputs
x1=2
x2=3
#weights
w1=0.5
w2=0.8
#bias
b=2
#weighted_sum = (x1 * w1) + (x2 * w2) + b
weighted_sum=(x1*w1)+(x2*w2)+b
print(f"Weighted sum: {weighted_sum}")

#-------------------------------------------- Liner Activation Function
from activationfunctions import linear

linear_output = linear(weighted_sum)
linear_function = print(f"Linear Activation Output: {linear_output}")

#-------------------------------------------- Binary Step Activation Function

from activationfunctions import binary_step
binary_step_function = print(f"Binary Step Activation Output: {binary_step(weighted_sum)}")

#-------------------------------------------- Sigmoid Activation Function

from activationfunctions import sigmoid

sigmoid_function = print(f"Sigmoid Activation Output: {sigmoid(weighted_sum)}")

#-------------------------------------------- Tanh Activation Function

from activationfunctions import tanh
tanh_function = print(f"tanh activation Output: {tanh(weighted_sum)}")

#-------------------------------------------- Relu Activation Function

from activationfunctions import Relu
relu_function = print(f"Relu Activation Output: {Relu(weighted_sum)}")