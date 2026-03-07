import numpy as np

# --------------------------------------------#
#                      SGD
# --------------------------------------------#
class SGD:
    """
    W_new = W_old - lr*grad_W
    b_new = b_old - lr*grad_b
    """
    def __init__(self, learning_rate: float, weight_decay: float = 0.0, **kwargs):
        self.lr           = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    def __repr__(self):
        return f"SGD(lr={self.lr}, wd={self.weight_decay})"
# -----------------------------------------------------------#
#                          Momentum
# -----------------------------------------------------------#

class Momentum:
    """
    v_W_new = beta*v_W_old + grad_W
    W_new = W_old - lr*v_W_new
    """
    def __init__(self, learning_rate: float, beta: float = 0.9, weight_decay: float = 0.0, **kwargs):
        self.lr           = learning_rate
        self.beta         = beta
        self.weight_decay = weight_decay
        self._v_W = []
        self._v_b = []
        self._initialised = False

    def _init_state(self, layers):
        self._v_W = [np.zeros_like(l.W) for l in layers]
        self._v_b = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            self._v_W[i] = self.beta * self._v_W[i] + layer.grad_W
            self._v_b[i] = self.beta * self._v_b[i] + layer.grad_b
            layer.W -= self.lr * self._v_W[i]
            layer.b -= self.lr * self._v_b[i]

    def __repr__(self):
        return f"Momentum(lr={self.lr}, beta={self.beta})"
# -------------------------------------------------------------------------------------#
#                        NAG  (Nesterov Accelerated Gradient)
# -------------------------------------------------------------------------------------#
class NAG:
    """
    Lookahead: W_look = W_old - beta x v_W_old
    Update velocity and weights:
        v_W_new = beta*v_W + grad_W_Look
        W_new   = W_old - lr*v_W_new
    """
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9,
                 weight_decay: float = 0.0, **kwargs):
        self.lr           = learning_rate
        self.beta         = beta
        self.weight_decay = weight_decay
        self._v_W = []
        self._v_b = []
        self._initialised = False

    def _init_state(self, layers):
        self._v_W = [np.zeros_like(l.W) for l in layers]
        self._v_b = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def apply_lookahead(self, layers):
        #Temporarily moves the weights to the lookahead position.
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            layer.W -= self.beta * self._v_W[i]
            layer.b -= self.beta * self._v_b[i]

    def undo_lookahead(self, layers):
        """Restores the weights from the lookahead position."""
        for i, layer in enumerate(layers):
            layer.W += self.beta * self._v_W[i]
            layer.b += self.beta * self._v_b[i]

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            self._v_W[i] = self.beta * self._v_W[i] + layer.grad_W
            self._v_b[i] = self.beta * self._v_b[i] + layer.grad_b
            layer.W -= self.lr * self._v_W[i]
            layer.b -= self.lr * self._v_b[i]

    # The NeuralNetwork training loop must call apply_lookahead() before forward-backward and undo_lookahead() before calling update().
    def __repr__(self):
        return f"NAG(lr={self.lr}, beta={self.beta})"
# ---------------------------------------------------------#
#                           RMSProp
# ---------------------------------------------------------#

class RMSProp:
    """
        s_W = rho*s_W + (1 - rho)*grad_W²
        W   = W - (lr / ((s_W + epsilon)^0.5))*grad_W
    """
    def __init__(self, learning_rate: float, rho: float = 0.9, epsilon: float = 1e-8, weight_decay: float = 0.0, **kwargs):
        self.lr           = learning_rate
        self.rho          = rho
        self.epsilon      = epsilon
        self.weight_decay = weight_decay
        self._s_W = []
        self._s_b = []
        self._initialised = False

    def _init_state(self, layers):
        self._s_W = [np.zeros_like(l.W) for l in layers]
        self._s_b = [np.zeros_like(l.b) for l in layers]
        self._initialised = True

    def update(self, layers):
        if not self._initialised:
            self._init_state(layers)
        for i, layer in enumerate(layers):
            self._s_W[i] = (self.rho * self._s_W[i]
                            + (1.0 - self.rho) * layer.grad_W ** 2)
            self._s_b[i] = (self.rho * self._s_b[i]
                            + (1.0 - self.rho) * layer.grad_b ** 2)
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self._s_W[i]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self._s_b[i]) + self.epsilon)

    def __repr__(self):
        return f"RMSProp(lr={self.lr}, rho={self.rho}, eps={self.epsilon})"
# ---------------------------------------------------------------------------#
#                                 Dispatch
# ---------------------------------------------------------------------------#
OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
}

def get_optimizer(name: str, **kwargs):
    """
    name   : 'sgd' or 'momentum' or 'nag' or 'rmsprop'
    """
    key = name.lower()
    return OPTIMIZERS[key](**kwargs)