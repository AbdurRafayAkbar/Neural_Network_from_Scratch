x1=2
x2=4
#hidden nueron 1 weights
w1_1=0.5
w1_2=0.2
b1=0.1
#hidden nueron2 weights
w2_1=1.5
w2_2=1.2
b2=0.1

z1=(x1*w1_1+x2*w1_2)+b1
print(f"Hidden Nueron 1  {z1}")

z2=(x1*w2_1+x2*w2_2)+b2
print(f"Hidden second Nueron {z2}")

#Apply Relu
def Relu(x):
    return max(0,x)

a1=Relu(z1)
a2=Relu(z2)

#Output Neuron now 
#we will send a1 and a2 values to find output since there are 2 nuerons its get 2 weights assigned

w_out1=0.4
w_out2=0.8

b_out=0.1

#Calculate Output
output=(
    (a1*w_out1+a2*w_out2)+b_out
)
print(f"Output is = {output}")