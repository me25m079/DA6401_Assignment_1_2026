import numpy as np
from ann.activations import get_activation, get_derivative

class NeuralLayer:
    """
    A single fully-connected layer.
        self.Z      - pre-activation linear combination 
        self.A      - post-activation output        
        self.grad_W - gradient w.r.t. W 
        self.grad_b - gradient w.r.t. b 
    """

    def __init__(self, n_in, n_out, activation, weight_init, is_output = False):
        """
        n_in  : number of input neurons
        n_out : number of output neurons
        activation   : 'relu' or 'sigmoid' or 'tanh' (ignored for output layer)
        weight_init  : 'random' or 'xavier'
        is_output    : True implies linear layer (no activation applied)
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation   = activation
        self.is_output    = is_output

        # Initialising the weights and biases.
        self.W, self.b = self._init_weights(weight_init)

        # Cache will be filled during forward pass.
        self.A_prev = None   # input to this layer   (batch, n_in)
        self.Z      = None   # pre-activation        (batch, n_out)
        self.A      = None   # post-activation       (batch, n_out)

        # Gradients will be filled during backward pass.
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        # Whether to use softmax or not.
        if not is_output:
            self._act_fn = get_activation(activation)
            self._act_deriv = get_derivative(activation)
    # --------------------------------------------------------------------------#
    #                             Weight initialisation
    # --------------------------------------------------------------------------#
    def _init_weights(self, method):
        if method == "xavier":
            limit = np.sqrt(6.0 / (self.n_in + self.n_out))
            W = np.random.uniform(-limit, limit, (self.n_in, self.n_out))
        elif method == "random":
            # Small random normal weights
            W = np.random.randn(self.n_in, self.n_out) * 0.01
        b = np.zeros((1, self.n_out)) #All biases are zeroes intially.
        return W, b
    # -------------------------------------------------------------------------#
    #                             Forward pass
    # -------------------------------------------------------------------------#
    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Compute Z = A_prev @ W + b  and  A = activation(Z).
        For the output (linear) layer, A = Z (no activation).
        
        A_prev : ndarray, shape (batch, n_in) ----->Returns----> A : ndarray, shape (batch, n_out)
        """
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        if self.is_output:
            self.A = self.Z  # linear – return raw logits
        else:
            self.A = self._act_fn(self.Z)
        return self.A
    # -------------------------------------------------------------------------#
    #                              Backward pass
    # -------------------------------------------------------------------------#
    def backward(self, dA: np.ndarray, weight_decay = 0.0) -> np.ndarray:
        """
        Computes gradients for this layer and returns delta to the previous layer.
        For hidden layers: dA is dL/dA (gradient of loss w.r.t. this layer's output).
        For the output layer: dA is already dL/dZ (combined loss and softmax gradient), so we skip the activation derivative step (is_output=True).
        weight_decay : L2 regularisation coefficient Lambda
        This returns dA_prev : gradient to pass to the previous layer.
        """
        batch_size = self.A_prev.shape[0]
        # dL/dZ: multiply by activation derivative for hidden layers and skip for output layer.
        if self.is_output:
            dZ = dA  # gradient already w.r.t. Z
        else:
            dZ = dA * self._act_deriv(self.Z)
        # Gradients w.r.t. W & b
        self.grad_W = (self.A_prev.T @ dZ)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        # L2 regularisation on weights (not biases).
        if weight_decay > 0.0:
            self.grad_W += weight_decay * self.W
        # Propagates gradient to the previous layer.
        dA_prev = dZ @ self.W.T
        return dA_prev
    # ------------------------------------------------------------------------------------#
    #                             Weight Serialisation
    # ------------------------------------------------------------------------------------#
    def get_weights(self) -> dict:
        """Return a copy of this layer's parameters."""
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights: dict):
        """Load parameters from a dict produced by get_weights()."""
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()
    # ------------------------------------------------------------------------------------------------------#
    def __repr__(self):
        act = "No Activation" if self.is_output else self.activation #Indicates whether the output layer is activated or not.
        return (f"NeuralLayer(in={self.n_out}, out={self.n_out}, "f"activation={act})")
