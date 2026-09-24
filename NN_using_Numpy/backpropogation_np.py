import numpy as np


# -----------------------------
# Training data
# -----------------------------

X = np.array([
    [2, 3],
    [4, 5],
    [6, 7]
])

y = np.array([
    [10],
    [20],
    [30]
])


# -----------------------------
# Hidden Layer
# -----------------------------

W1 = np.array([
    [0.5, 0.3, 0.1],
    [0.2, 0.4, 0.5]
])

b1 = np.array([0.1, 0.1, 0.1])


# -----------------------------
# Output Layer
# -----------------------------

W2 = np.array([
    [0.4],
    [0.3],
    [0.2]
])

b2 = np.array([0.1])


learning_rate = 0.001


# -----------------------------
# Training
# -----------------------------

for epoch in range(10000):

    # ===== Forward Propagation =====

    Z1 = X @ W1 + b1

    A1 = np.maximum(0, Z1)

    Z2 = A1 @ W2 + b2

    y_pred = Z2


    # ===== Loss =====

    loss = np.mean((y_pred - y) ** 2)


    # ===== Backpropagation =====

    # Output layer
    dZ2 = 2 * (y_pred - y) / y.size

    dW2 = A1.T @ dZ2

    db2 = np.sum(dZ2, axis=0)


    # Hidden layer
    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1

    db1 = np.sum(dZ1, axis=0)


    # ===== Update =====

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", loss)


print("\nPredictions:")
print(y_pred)

print("\nActual:")
print(y)