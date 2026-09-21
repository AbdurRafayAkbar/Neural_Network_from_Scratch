def linear(x):
    return x

def binary_step(x):
    return 1 if x > 0 else 0

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

def tanh(x):
    import math
    e_pos = math.exp(x)
    e_neg = math.exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def Relu(x):
    return x if x > 0 else 0

def Relu_derivative(x):
    return 1 if x > 0 else 0

if __name__ == "__main__":
    values = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    for value in values:
        print(f"Linear({value}) = {linear(value)}")
    for value in values:
        print(f"Binary Step({value}) = {binary_step(value)}")
    for value in values:
        print(f"Sigmoid({value}) = {sigmoid(value)}")
    for value in values:
        print(f"Tanh({value}) = {tanh(value)}")
    for value in values:
        print(f"Relu({value}) = {Relu(value)}")