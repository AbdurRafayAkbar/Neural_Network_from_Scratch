#Inputs
x=2
target=10
weight=2
bias=0
learning_rate=0.1
#Prediction
prediction=(x*weight)+bias
#find the Error
error=target-prediction
#Calculate The Loss
loss=error**2

#Calculate_gradients
 
weight_gradient=2 * (prediction-target) * x
bias_gradient=2 * (prediction-target)

#Update_weights
weight=weight-(learning_rate*weight_gradient)
bias=bias-(learning_rate*bias_gradient)

print(f"Prediction: {prediction}")
print(f"Error: {error}")
print(f"Loss: {loss}")  
print(f"Weight Gradient: {weight_gradient}")
print(f"Bias Gradient: {bias_gradient}")
print(f"Updated Weight: {weight}")
print(f"Updated Bias: {bias}")