import numpy as np
# ---------------------------------------------------------------------------#
#                           Activation functions
# ---------------------------------------------------------------------------#
def sigmoid(Z):
    """
    g(z) = 1 / (1 + e^(-z)).
    Clipped for numerical stability. (To avoid overflow and vanishing gradients.)  
    """
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z):
    """ dg/dz = g(z) · (1 - g(z)) """
    s = sigmoid(Z)
    return s * (1.0 - s)

def tanh(Z):
    """Hyperbolic tangent activation."""
    return np.tanh(Z)

def tanh_derivative(Z):
    """dtanh/dz = 1 - tanh²(z)"""
    return 1.0 - np.tanh(Z) ** 2

def relu(Z):
    """g(z) = max(0, z)"""
    return np.maximum(0.0, Z)

def relu_derivative(Z):
    """dg/dz = 1 if z > 0 else 0."""
    return (Z > 0).astype(float)

def softmax(Z):
    """
    Softmax activation (numerically stable).
    Applied row-wise: each row is one sample.
    Parameters
    ----------
    Z : ndarray, shape (batch, C)

    Returns
    -------
    P : ndarray, shape (batch, C)  - probability distribution over C classes
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)   # subtract row-max for stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
# ---------------------------------------------------------------------------#
#                                 Dispatch
# ---------------------------------------------------------------------------#
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh":    tanh,
    "relu":    relu,
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "tanh":    tanh_derivative,
    "relu":    relu_derivative,
}


def get_activation(name: str):
    """Returns the Activation Function by name(case-insensitive)."""
    key = name.lower()
    return ACTIVATIONS[key]

def get_derivative(name: str):
    """Returns the Activation Function's derivative by name(case-insensitive)."""
    key = name.lower()
    return DERIVATIVES[key]