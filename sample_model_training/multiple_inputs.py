import numpy as np
X=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]],dtype=float)

Y=np.array([
    [25],
    [50],
    [75]
],dtype=float)

weights=np.zeros(3)
bias=0.0

epoch=1000
learning_rate=0.01

for epoch in range(epoch):
    total_loss=0
    for inputs,targets in zip(X,Y):

        #forward pass
        prediction=np.dot(inputs,weights)+bias

        #loss calculation
        loss=(prediction-targets)**2

        total_loss+=loss