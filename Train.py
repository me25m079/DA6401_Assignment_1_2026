import numpy as np
import argparse
import json
import wandb
from keras.datasets import mnist, fashion_mnist

from Models.Activations import get_activation
from Models.Initializers import get_initializer
from Utils.Losses import get_loss
from Models.Optimizers import get_optimizer
from Utils.Metrics import confusion_matrix, accuracy, precision_recall_f1


# ---------------- Layers ----------------
class Dense:
    def __init__(self, in_dim, out_dim, initializer):
        self.W = initializer(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.grad_W = np.dot(self.x.T, dout)
        self.grad_b = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)


class MLP:
    def __init__(self, input_dim, hidden_sizes, output_dim, activation, initializer):
        self.layers = []
        dims = [input_dim] + hidden_sizes + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(Dense(dims[i], dims[i + 1], initializer))
            if i < len(dims) - 2:
                self.layers.append(activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)


# ---------------- Utils ----------------
def one_hot(y, num_classes):
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y] = 1
    return oh


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_dataset(name):
    if name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset")

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    return X_train, y_train, X_test, y_test


def evaluate(model, X, y):
    logits = model.forward(X)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)

    cm = confusion_matrix(y, preds, num_classes=10)
    acc = accuracy(y, preds)
    prec, rec, f1 = precision_recall_f1(cm)

    return acc, prec, rec, f1


# ---------------- Training ----------------
def train(args):
    wandb.init(project="mlp-numpy", config=vars(args))
    config = wandb.config

    X_train, y_train, X_test, y_test = load_dataset(config.dataset)

    initializer = get_initializer(config.weight_init)
    activation = get_activation(config.activation)
    loss_fn = get_loss(config.loss)
    optimizer = get_optimizer(config.optimizer)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_sizes=config.hidden_size,
        output_dim=10,
        activation=activation,
        initializer=initializer
    )

    best_f1 = -1.0

    for epoch in range(config.epochs):
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        epoch_loss = 0.0

        for i in range(0, len(X_train), config.batch_size):
            xb = X_train[i:i + config.batch_size]
            yb = y_train[i:i + config.batch_size]

            logits = model.forward(xb)

            if config.loss == "mse":
                yb_oh = one_hot(yb, 10)
                loss = loss_fn.forward(logits, yb_oh)
                dout = loss_fn.backward()
            else:
                loss = loss_fn.forward(logits, yb)
                dout = loss_fn.backward()

            model.backward(dout)

            for layer in model.layers:
                if hasattr(layer, "W"):
                    optimizer.update(layer, config.learning_rate, config.weight_decay)

            epoch_loss += loss

        epoch_loss /= (len(X_train) // config.batch_size)

        test_acc, test_prec, test_rec, test_f1 = evaluate(model, X_test, y_test)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1
        })

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Test F1: {test_f1:.4f}"
        )

        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1

            weights = []
            for layer in model.layers:
                if hasattr(layer, "W"):
                    weights.append({"W": layer.W, "b": layer.b})

            np.save("best_model.npy", weights)

            with open("best_config.json", "w") as f:
                json.dump(dict(config), f, indent=4)

            wandb.save("best_model.npy")
            wandb.save("best_config.json")

    wandb.finish()


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-l", "--loss", choices=["mse", "cross_entropy"], required=True)
    parser.add_argument("-o", "--optimizer", required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)
    parser.add_argument("-a", "--activation", required=True)
    parser.add_argument("-wi", "--weight_init", required=True)

    args = parser.parse_args()
    train(args)