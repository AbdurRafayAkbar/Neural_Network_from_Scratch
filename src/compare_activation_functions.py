from activationfunctions import linear, binary_step, tanh, Relu, sigmoid

print("Input | Linear | Binary | Tanh | ReLU |sigmoid")
values=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
for value in values:
    print(f"{value} | {linear(value):6.2f} | {binary_step(value):6.2f} | {tanh(value):6.2f} | {Relu(value):6.2f} | {sigmoid(value):6.2f}")