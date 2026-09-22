import numpy as np
#1d array
x_1=np.array([2,3,4,5,6,7])

print(F" 1d array shape = {x_1.shape}")


#2d array

x_2=np.array([[2,4,6,8],
            [1,3,5,7]])

print(f"2d array shape = {x_2.shape}")

#Element Ooperations

print(f"multiply by 2 2d array{x_2*2}")

#addition operation

a=np.array([2,4,6,8])
b=np.array([10,20,30,40])
print(f" added a + b ={(a+b)}")

#dot product

print(f"Multiplied a . b = {np.dot(a,b)}")