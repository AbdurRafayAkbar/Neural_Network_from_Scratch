#inputs

x_values=[1,2,3,4,5]
target_values=[3,5,7,9,11]

#Learning rate
learning_rate=0.01
epochs=100

#weights and bias

weight=0
bias=0

#Training the model

for epoch in range(epochs):
    total_loss=0
    for x,target in zip(x_values,target_values):
        #forward Prediction
        prediction=(x*weight)+bias
        #Calculate loss
        loss=(target-prediction)**2

        total_loss = total_loss + loss

        #Calculate gradients

        weight_gradient=2*(prediction-target)*x
        bias_gradient=2*(prediction-target)

        #update weights and bias

        weight=weight-(learning_rate*weight_gradient)
        bias=bias-(learning_rate*bias_gradient)

    print(f"Epoch {epoch+1}: Total Loss: {total_loss}, Weight: {weight}, Bias: {bias}")


