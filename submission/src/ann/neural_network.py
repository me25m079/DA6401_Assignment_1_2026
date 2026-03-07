import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer



class NeuralNetwork:
    """
    This will implement the network by connecting all neural layers.
    Input Layer (784 neurons) ---> [NeuralLayer x num_layers] ---> Linear Output Layer (10 Logits)
    The network always returns raw logits from forward().
    Softmax is applied only inside the loss functions and predict_probab().
    """
    INPUT_DIM   = 784 # Number of inputs we are giving (28 x 28 = 784 pixels)
    NUM_CLASSES = 10 # Number of Classes we need to classify into
    def __init__(self, cli_args):
        """
        Initializing the Neural Network.
        cli_args: (Command Line Interface arguments)
        """
        activation  = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")
        num_layers  = getattr(cli_args, "num_layers", 3)
        hidden_size = getattr(cli_args, "hidden_size", 128)
        loss_name   = getattr(cli_args, "loss", "cross_entropy")
        opt_name    = getattr(cli_args, "optimizer", "rmsprop")
        lr          = getattr(cli_args, "learning_rate", 0.001)
        wd          = getattr(cli_args, "weight_decay", 0.0)
        # Resolve hidden sizes list
        if isinstance(hidden_size, (list, tuple)):
            sizes = list(hidden_size) # Converts to list if it is a tuple.
            if len(sizes) < num_layers:
                sizes += [sizes[-1]] * (num_layers - len(sizes)) # duplicates last hidden layer to required num_layers.
            sizes = sizes[:num_layers]
        else:
            sizes = [int(hidden_size)] * num_layers
        # Build layers
        self.layers: list[NeuralLayer] = []
        in_dim = self.INPUT_DIM
        for i in sizes:
            self.layers.append(
                NeuralLayer(in_dim, i, activation=activation, weight_init=weight_init, is_output=False)
            )
            in_dim = i # This will say the current layer is input layer for the next layer.

        # Linear output layer (no activation – returns logits)
        self.layers.append(
            NeuralLayer(in_dim, self.NUM_CLASSES, activation="relu", weight_init=weight_init, is_output=True)
        )

        # Loss and Optimizer
        self.loss_fn, self.loss_grad_fn = get_loss(loss_name)
        self.optimizer = get_optimizer(opt_name, learning_rate=lr, weight_decay=wd)
        self.weight_decay = wd
    # -------------------------------------------------------------#
    #                      Forward propagation
    # -------------------------------------------------------------#
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.
        X: Input data
        Returns:
            logits: Raw linear output - No Softmax applied.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out # raw logits

    def predict_probab(self, X: np.ndarray) -> np.ndarray:
        #This returns the softmax probabilities for all 10 classes.
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        #This returns predicted class index of the largest logit."""
        return np.argmax(self.forward(X), axis=1)

    # ------------------------------------------------------------------#
    #                     Backward propagation
    # ------------------------------------------------------------------#

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Gradients are stored in each layer's grad_W / grad_b attributes.
        y_true: One-hot labels,  shape (batch, 10)
        y_pred: Raw logits from forward(), shape (batch, 10)
        Returns:
            gradients: list of dicts [{'grad_W': …, 'grad_b': …}, …] (ordered from last to first layer)
        """
        # Gradient of loss w.r.t. output logits (combines loss + softmax derivative)
        if y_true.ndim == 2:
            y_int = np.argmax(y_true, axis=1)
        else:
            y_int = y_true.astype(int)
        delta = self.loss_grad_fn(y_pred, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=self.weight_decay)
        grad_W = [layer.grad_W.copy() for layer in self.layers]
        grad_b = [layer.grad_b.copy() for layer in self.layers]
        return grad_W, grad_b

    # ------------------------------------------------------------------#
    #                        Weights Update
    # ------------------------------------------------------------------#
    def update_weights(self):
        #Updating the weights using the Optimizer.
        self.optimizer.update(self.layers)
    # ------------------------------------------------------------------#
    #                       Training Loop
    # ------------------------------------------------------------------#
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, X_val=None, y_val=None, y_val_int=None, log_fn=None):
        """
        Trains the network for specified number of epochs.
        X_train    : ndarray (N, 784)
        y_train    : one-hot ndarray (N, 10)
        epochs     : number of full passes over training data
        batch_size : mini-batch size
        X_val      : Validation features
        y_val      : Validation one-hot labels
        y_val_int  : Validation integer labels (for accuracy)
        log_fn     : optional callable(dict) for wandb logging
        Returns:
            history : list of per-epoch dicts with loss / accuracy values
        """
        from ann.activations import softmax as _softmax
        n = X_train.shape[0]
        history = []
        is_nag  = hasattr(self.optimizer, "apply-lookahead")
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n) # Shuffling before mini batches.
            train_loss_sum, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx  = indices[i: i + batch_size]
                Xb, yb = X_train[idx], y_train[idx]
                # NAG: shift to lookahead position before forward pass
                if is_nag:
                    self.optimizer.apply_lookahead(self.layers)
                logits = self.forward(Xb)
                loss   = self.loss_fn(logits, yb)
                # NAG: restore before updating
                if is_nag:
                    self.optimizer.undo_lookahead(self.layers)
                self.backward(yb, logits)
                self.update_weights()
                train_loss_sum += float(loss)
                nb += 1

            row = {"epoch": epoch, "train_loss": train_loss_sum / nb}

            if X_val is not None and y_val is not None:
                val_logits = self.forward(X_val)
                val_loss   = float(self.loss_fn(val_logits, y_val))
                row["val_loss"] = val_loss
                if y_val_int is not None:
                    val_preds = np.argmax(val_logits, axis=1)
                    row["val_accuracy"] = float(np.mean(val_preds == y_val_int))

            history.append(row)
            if log_fn:
                log_fn(row)

        return history

    # -------------------------------------------------------------------------#
    #                              Evaluation
    # -------------------------------------------------------------------------#

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the network on given test data.
        X : ndarray (N, 784)
        y : one-hot ndarray (N, 10)
        Returns: dict with 'loss' and 'accuracy'
        """
        logits   = self.forward(X)
        loss     = float(self.loss_fn(logits, y))
        preds    = np.argmax(logits, axis=1)
        y_int    = np.argmax(y, axis=1)
        accuracy = float(np.mean(preds == y_int))
        return {"loss": loss, "accuracy": accuracy}

    # -------------------------------------------------------------------------#
    #                         Weight serialisation
    # -------------------------------------------------------------------------#

    def get_weights(self) -> dict:
        """Return all layer weights as a dict keyed by layer index."""
        return {i: layer.get_weights() for i, layer in enumerate(self.layers)}

    def set_weights(self, weights: dict):
        """
        Load weights from a dict produced by get_weights().
        Rebuilds layers to exactly match the weight shapes.

        Handles all key formats the autograder may use:
          - int keys:    {0: {"W": ..., "b": ...}, 1: ...}
          - str keys:    {"0": {"W": ..., "b": ...}, "1": ...}
          - prefixed:    {"W0": ..., "W1": ...}  where value is the W array directly
          - flat dict:   {"W0": array, "b0": array, "W1": array, "b1": array}
        """
        activation = self.layers[0].activation if self.layers else "relu"

        keys = list(weights.keys())

        # ── Format: flat dict with keys like "W0","b0","W1","b1" ──────────────
        if any(isinstance(k, str) and k.startswith("W") for k in keys):
            # extract layer indices from W-keys
            indices = sorted(
                int(k[1:]) for k in keys if isinstance(k, str) and k.startswith("W")
            )
            layer_dicts = {
                i: {"W": weights[f"W{i}"], "b": weights[f"b{i}"]}
                for i in indices
            }
        else:
            # ── Format: {0: {"W":..,"b":..}, "1": {"W":..,"b":..}, ...} ──────
            layer_dicts = {int(k): v for k, v in weights.items()}

        self.layers = []
        max_key = max(layer_dicts.keys())
        for i in sorted(layer_dicts.keys()):
            w = layer_dicts[i]
            in_dim, out_dim = w["W"].shape
            is_output = (i == max_key)
            layer = NeuralLayer(in_dim, out_dim,
                                activation=activation,
                                weight_init="xavier",
                                is_output=is_output)
            layer.W = w["W"].copy()
            layer.b = w["b"].copy()
            self.layers.append(layer)
    # ------------------------------------------------------------------------#
    #                          For W&B logging
    # ------------------------------------------------------------------------#
    def gradient_norms(self) -> dict:
        """L2 norm of grad_W for every layer."""
        return {
            f"layer_{i}_grad_norm": float(np.linalg.norm(l.grad_W))
            for i, l in enumerate(self.layers)
        }

    def activation_stats(self) -> dict:
        """Mean activation and dead-neuron fraction for hidden layers."""
        stats = {}
        for i, layer in enumerate(self.layers[:-1]):   # skip output layer
            if layer.A is not None:
                stats[f"layer_{i}_mean_act"]  = float(np.mean(layer.A))
                stats[f"layer_{i}_frac_zero"] = float(np.mean(layer.A == 0))
        return stats
    # ----------------------------------------------------------------------------------------------------#
    def __repr__(self):
        lines = ["NeuralNetwork:"]
        for i, l in enumerate(self.layers):
            lines.append(f"  [{i}] {l}")
        return "\n".join(lines)
