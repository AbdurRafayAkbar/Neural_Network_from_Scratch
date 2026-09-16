import numpy as np
X=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]],dtype=float)

Y = np.array([25, 50, 75], dtype=float)  # Flat 1D array


weights=np.zeros(3)
bias=0.0

epoch=1000
learning_rate=0.0001

for epoch in range(epoch):
    total_loss=0
    for inputs,targets in zip(X,Y):

        #forward pass
        prediction=np.dot(inputs,weights)+bias

        #loss calculation
        loss=(prediction-targets)**2

        total_loss+=loss

        #gradient calculation

        weight_gradient=2*(prediction-targets)*inputs
        bias_gradient=2*(prediction-targets)

        #update weights and bias

        weights=weights-(learning_rate*weight_gradient)
        bias=bias-(learning_rate*bias_gradient)

    average_loss=total_loss/len(X)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Loss = {average_loss:.3f}, "  # No [0] needed!
            f"Weights = {weights}, "
            f"Bias = {bias:.3f}"            # No [0] needed!
        )



print("\nFinal parameters:")
print("Weights:", weights)
print("Bias:", bias)