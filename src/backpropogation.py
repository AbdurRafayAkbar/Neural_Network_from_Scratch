# understanding basic backpropagation algorithm
from activationfunctions import Relu
#inputs

x=2
y=10
w=3
b=1
learning_rate=0.01

#Forward pass

predicted=(w*x)+b

#Activation function (ReLU)

a=Relu(predicted)
print(a)
#loss function (Mean Squared Error)
loss=(predicted-y)**2
print(loss)

