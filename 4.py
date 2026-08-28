import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)

b1 = np.zeros((1, 4))
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for _ in range(10000):

    h = sigmoid(
        X @ W1 + b1
    )

    o = sigmoid(
        h @ W2 + b2
    )

    e = y - o

    d2 = e * o * (1 - o)

    d1 = (
        d2 @ W2.T
    ) * h * (1 - h)

    W2 += (
        h.T @ d2
    ) * 0.1

    W1 += (
        X.T @ d1
    ) * 0.1

    b2 += (
        d2.sum(axis=0)
    ) * 0.1

    b1 += (
        d1.sum(axis=0)
    ) * 0.1

print(
    (o >= 0.5).astype(int)
)
