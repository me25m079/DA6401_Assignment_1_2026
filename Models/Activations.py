import numpy as np


class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


def get_activation(name):
    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "relu":
        return ReLU()
    else:
        raise ValueError("Unsupported activation")