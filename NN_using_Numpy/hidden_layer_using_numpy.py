import numpy as np

x=np.array([
    [1,2],
    [3,4],
    [5,6]

])
print(f"x shape : {x.shape}")

#3 hidden Layer Neurons

w1=np.array([[1,2,3],
             [4,5,6]])

bias=np.array([0.1,0.1,0.1])

#output layer Neuron weights

w1_out=np.array([
    [3],
    [5],
    [8]
])
print(f"w1_out shape : {w1_out.shape}")
b_out=np.array([0.1])

    #FORWARD Prop
z1=(x@w1)+bias
print(f"z1 shape = {z1.shape}")
print(z1)

    #RELU

z2=(z1@w1_out)+b_out
y_pred=z2
print(f"forward {z1}")
print(f"prediction {z2}")