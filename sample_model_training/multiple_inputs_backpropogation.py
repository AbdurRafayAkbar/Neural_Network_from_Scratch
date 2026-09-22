# Inputs
x1 = 2
x2 = 4
x3 = 6

# Weights
w1 = 3
w2 = 5
w3 = 7

# Bias
b = 5

# Learning rate
lr = 0.01

# Target
y = 20


# =========================
# FORWARD PASS
# =========================

z = x1 * w1 + x2 * w2 + x3 * w3 + b

# ReLU
if z > 0:
    prediction = z
else:
    prediction = 0

# Loss
loss = (prediction - y) ** 2


print("FORWARD PASS")
print("z =", z)
print("Prediction =", prediction)
print("Loss =", loss)


# =========================
# BACKPROPAGATION
# =========================

# Loss gradient
dL_da = 2 * (prediction - y)

# ReLU derivative
if z > 0:
    da_dz = 1
else:
    da_dz = 0

# Gradient flowing into z
dL_dz = dL_da * da_dz

# Weight gradients
dL_dw1 = dL_dz * x1
dL_dw2 = dL_dz * x2
dL_dw3 = dL_dz * x3

# Bias gradient
dL_db = dL_dz


print("\nBACKPROPAGATION")
print("dL/da =", dL_da)
print("da/dz =", da_dz)
print("dL/dz =", dL_dz)

print("dL/dw1 =", dL_dw1)
print("dL/dw2 =", dL_dw2)
print("dL/dw3 =", dL_dw3)
print("dL/db =", dL_db)


# =========================
# GRADIENT DESCENT
# =========================

w1 = w1 - lr * dL_dw1
w2 = w2 - lr * dL_dw2
w3 = w3 - lr * dL_dw3
b = b - lr * dL_db


print("\nUPDATED PARAMETERS")
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("b =", b)