#INPUTS
#we have 3 nuerons, 2input features, 3 training examples
import numpy as np
x=np.array([[1,2],  #shape(3x2)
            [3,4],
            [5,6]])

Y=np.array([[10,20,30], #shape(3x3)
            [40,50,60],
            [70,80,90]])

w=np.array([[0.1,0.2,0.3],  #shape(2x3)
            [0.4,0.5,0.6]])

b=np.array([[0.1,0.1,0.1]]) #shape(1x3)