import numpy as np

def binary_step(x):
    if x >= 0:
        return 1
    else:
        return 0

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (
        np.exp(x) - np.exp(-x)
    ) / (
        np.exp(x) + np.exp(-x)
    )

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.01 * x

def elu(x, alpha=1):
    if x > 0:
        return x
    else:
        return alpha * (np.exp(x) - 1)

def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

x = -2

print("Input:", x)
print("Binary Step:", binary_step(x))
print("Linear:", linear(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("ELU:", elu(x))

values = np.array([1, 2, 3])

print("Softmax:", softmax(values))
