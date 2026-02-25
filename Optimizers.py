import numpy as np


class SGD:
    def update(self, layer, lr, wd):
        layer.W -= lr * (layer.grad_W + wd * layer.W)
        layer.b -= lr * layer.grad_b


class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = {}

    def update(self, layer, lr, wd):
        if layer not in self.v:
            self.v[layer] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            }

        self.v[layer]["W"] = self.beta * self.v[layer]["W"] + layer.grad_W
        self.v[layer]["b"] = self.beta * self.v[layer]["b"] + layer.grad_b

        layer.W -= lr * (self.v[layer]["W"] + wd * layer.W)
        layer.b -= lr * self.v[layer]["b"]


class RMSProp:
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, layer, lr, wd):
        if layer not in self.s:
            self.s[layer] = {
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            }

        self.s[layer]["W"] = self.beta * self.s[layer]["W"] + (1 - self.beta) * layer.grad_W ** 2
        self.s[layer]["b"] = self.beta * self.s[layer]["b"] + (1 - self.beta) * layer.grad_b ** 2

        layer.W -= lr * (layer.grad_W / (np.sqrt(self.s[layer]["W"]) + self.eps) + wd * layer.W)
        layer.b -= lr * layer.grad_b / (np.sqrt(self.s[layer]["b"]) + self.eps)


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, lr, wd):
        if layer not in self.m:
            self.m[layer] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
            self.v[layer] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

        self.t += 1

        for param in ["W", "b"]:
            grad = getattr(layer, f"grad_{param}")
            self.m[layer][param] = self.beta1 * self.m[layer][param] + (1 - self.beta1) * grad
            self.v[layer][param] = self.beta2 * self.v[layer][param] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[layer][param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[layer][param] / (1 - self.beta2 ** self.t)

            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if param == "W":
                layer.W -= update + lr * wd * layer.W
            else:
                layer.b -= update


def get_optimizer(name):
    if name == "sgd":
        return SGD()
    elif name == "momentum":
        return Momentum()
    elif name == "rmsprop":
        return RMSProp()
    elif name == "adam":
        return Adam()
    else:
        raise ValueError("Unsupported optimizer")