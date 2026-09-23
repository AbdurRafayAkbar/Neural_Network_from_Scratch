#INPUTS
#we have 3 nuerons, 2input features, 3 training examples
import numpy as np
x=np.array([[1,2],  #shape(3x2)
            [3,4],
            [5,6]])

y=np.array([[10,20,30], #shape(3x3)
            [40,50,60],
            [70,80,90]])

w=np.array([[0.1,0.2,0.3],  #shape(2x3)
            [0.4,0.5,0.6]])

b=np.array([[0.1,0.1,0.1]]) #shape(1x3)

lr=0.001

for epoch in range(1000):
        
    #------------------Forward Pass
    z=(x@w)+b
    # print(z)

    #Relu Activation

    #------------------Z remain same as all values are positive

    #------------------Calculate the loss using MSE(Mean Squared Error)

    Loss=np.mean((y-z)**2)  #first y-z then its square and then Mean

    # print(Loss)

    #------------------BACK PROPOGATION

    dz=2*(z-y)/y.size #y.size bcz the error is mean and we divide it by total values

    #Weight Gradient

    dw=x.T@dz

    #Bias Gradient

    db=np.sum(dz,axis=0)

    #Update Weights

    w=w-(lr*dw)
    b=b-(lr*db)

    if epoch %10 == 0:
        print(f"epoch {epoch} loss is {Loss}")
