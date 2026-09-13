# In this we are comapring the performance of different activation functions in a neural network. The activation functions we will be comparing are ReLU, Sigmoid, and Tanh.
#  We will use a simple feedforward neural network architecture and evaluate the performance of each activation function on a classification task.
# we will use the same architecture and hyperparameters for each activation function to ensure a fair comparison.
# we will import the activation functions to nueron.py and check what our final result come


# Linear Activation Function
def linear(x):
    return x

values=[-5,-4,-3,-2,-1,0,1,2,3,4,5]

for value in values:
    print(f"Linear({value}) = {linear(value)}")

def binary_step(x):
    if x>0:
        return 1
    else:
        return 0

for value in values:
    print(f"Binary Step({value}) = {binary_step(value)}")

def sigmoid(x):
    import math
    return 1/(1+math.exp(-x))

for value in values:
    print(f"Sigmoid({value})= {sigmoid(value)}")

def tanh(x):
    import math
    return (math.exp(x)-math.exp(-x)/math.exp(x)+math.exp(-x))

for value in values:
    print(f"Tanh({value})= {tanh(value)}")
    