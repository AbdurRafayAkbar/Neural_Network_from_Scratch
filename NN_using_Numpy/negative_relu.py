x1=2
x2=-4 # change it to positive and find the difference

w1=0.1
w2=0.2
b=1

z=((x1*w1)+(x2*w2)+b)

def Relu(x):
    return max(0,x)
#Nuerons
z1=x1*0.3+x2*0.8+b
z2=x1*0.1+x2*0.2+b
z3=x1*0.9+x2*0.4+b

#apply Relu
a1=Relu(z1)
a2=Relu(z2)
a3=Relu(z3)

print(f"a1 = {a1}" )
print(f"a2 = {a2}" )
print(f"a3 = {a3}" )

# weights assigned for output value
w1_out=0.5
w2_out=0.8
w3_out=0.4
b=0.1
output=(
    (a1*w1_out+a2*w2_out+a3*w3_out)+b
)
print(f"output = {output}")