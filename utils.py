import numpy as np
import matplotlib.pyplot as plt


def encodeY(x):
    return np.array([[1 if j == i else 0 for j in range(10)] for i in x])


def imshow(im):
    plt.imshow(im.reshape((8, 8)), cmap="gray")
    plt.show()


def add_bias(array: np.ndarray) -> np.ndarray:
    return np.hstack((array, np.ones((array.shape[0], 1))))


def reLU(x):

    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(pred, y):
    epsilon = 1e-12
    pred = np.clip(pred, epsilon, 1.0 - epsilon)
    return -np.sum(y * np.log(pred)) / y.shape[0]


def get_cost_function(pred: np.ndarray, y: np.ndarray):
    assert pred.shape == y.shape, "Input arrays aren't the same size !"
    L, C = pred.shape
    return np.sum((pred - y) ** 2) / (2 * C)


def get_gradient_Y(pred: np.ndarray, y: np.ndarray):
    """
    partial derivative of J (Loss function) by Y hat (prediction)
    """
    assert pred.shape == y.shape, "Input arrays aren't the same size !"
    L, C = pred.shape
    return ((pred - y)) / (C)
    # return ((pred - y)) / (1)


def get_gradient_reLU(z: np.ndarray):
    """
    gradient of the reLU function
    """
    return (z > 0).astype(float)


def get_gradient_w3(a2):
    """
    partial derivative of z3 hat (prediction) by w3
    """
    return a2


def model(x, w1, w2, w3):
    z1 = add_bias(x).dot(w1)
    a1 = reLU(z1)

    z2 = add_bias(a1).dot(w2)
    a2 = reLU(z2)

    z3 = add_bias(a2).dot(w3)
    a3 = reLU(z3)

    return a3


def get_gradient_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def predict(X, w1, w2, w3):
    z1 = add_bias(X).dot(w1)
    a1 = reLU(z1)
    z2 = add_bias(a1).dot(w2)
    a2 = reLU(z2)
    z3 = add_bias(a2).dot(w3)
    a3 = softmax(z3)
    return np.argmax(a3, axis=1)
