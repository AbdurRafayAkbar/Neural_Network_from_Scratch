import numpy as np
x=np.array([
    [2,3],
    [4,5],
    [6,7]
])

y=np.array([
    [10],
    [20],
    [30]
])

#hidden Layer

w1=np.array([
    [0.5,0.7,0.9],
    [1.2,1.6,1.8]
])
b1=np.array([0.1,0.1,0.1])

# Output    Layer

w2=np.array([
    [0.4],
    [0.6],
    [0.8]
])
b_out=np.array([0.1])

lr=1e-2

#Forward Propgation
z1=(x@w1)+b1



