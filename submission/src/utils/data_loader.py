import numpy as np
from sklearn.model_selection import train_test_split

def _to_onehot(y, num_classes = 10): # One-Hot Encoding
    ohe = np.zeros((len(y), num_classes), dtype=np.float32)
    ohe[np.arange(len(y)), y] = 1.0
    return ohe

def load_dataset(name: str, val_split: float):
    """
    Loads MNIST or Fashion-MNIST using keras.datasets.
    """
    if name == "mnist":
        from keras.datasets import mnist
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()

    X_tr = X_tr.reshape(-1, 784).astype(np.float32) / 255.0
    X_te = X_te.reshape(-1, 784).astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_split, stratify=y_tr
    )
    return (
        (X_train, _to_onehot(y_train)),
        (X_val,   _to_onehot(y_val)),
        (X_te,    _to_onehot(y_te)),
        (y_train, y_val, y_te),
    )

def get_batches(X, y, batch_size, shuffle=True):
    n       = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]