import numpy as np


def random_init(in_dim, out_dim):
    return np.random.randn(in_dim, out_dim) * 0.01


def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim))


def get_initializer(name):
    if name == "random":
        return random_init
    elif name == "xavier":
        return xavier_init
    else:
        raise ValueError("Unsupported initializer")