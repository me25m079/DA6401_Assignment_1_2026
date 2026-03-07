import numpy as np
from ann.activations import softmax

def _to_int_labels(y_true: np.ndarray) -> np.ndarray:
    # Accepts one-hot (N,C) or integer/float (N,) — always returns int (N,)
    y = np.asarray(y_true)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y.astype(int)

# ---------------------------------------------------------------------------#
#                  Cross-Entropy Loss (paired with Softmax)
# ---------------------------------------------------------------------------#
def cross_entropy_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    L = -mean(sum(y_c*log(p_c))
    logits : raw linear outputs (Not softmax-activated)
    y_true : one-hot encoded labels
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    y_int = _to_int_labels(y_true)
    eps = 1e-12
    correct_log_probs = np.log(
        np.clip(probs[np.arange(batch_size), y_int], eps, 1.0)
    )
    return -np.mean(correct_log_probs)

def cross_entropy_grad(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    dL/dz = p_c - y_c
    """
    batch_size = logits.shape[0]
    y_int = _to_int_labels(y_true)
    probs = softmax(logits)
    probs[np.arange(batch_size), y_int] -= 1.0
    return probs / batch_size
    
# -------------------------------------------------------------------#
#                       Mean Squared Error Loss
# -------------------------------------------------------------------#
def mse_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    L = mean(sum(logit_c - y_c)^2)
    """
    y_int = _to_int_labels(y_true)
    y_one_hot = _one_hot(y_int, logits.shape[1])
    return np.mean((logits - y_one_hot) ** 2)

def mse_grad(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss w.r.t. logits:
        dL/dz = 2 * (logit_c - y_c)
    """
    batch_size, num_classes = logits.shape
    y_int = _to_int_labels(y_true)
    y_one_hot = _one_hot(y_int, num_classes)
    return 2.0 * (logits - y_one_hot) / (batch_size * num_classes)
# --------------------------------------------------------------------------#
#                                Dispatch
# --------------------------------------------------------------------------#
LOSSES = {
    "cross_entropy":      (cross_entropy_loss, cross_entropy_grad),
    "mean_squared_error": (mse_loss, mse_grad),
}

def get_loss(name: str):
    """
    Returns (loss_fn, grad_fn) for the given loss.
    name : 'cross_entropy' or 'mean_squared_error'
    """
    key = name.lower()
    return LOSSES[key]
