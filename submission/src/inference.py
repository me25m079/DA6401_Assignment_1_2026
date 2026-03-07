import argparse
import os
import sys
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(__file__)) # This is used to look in the src folder.

from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from utils.data_loader import load_dataset

VAL_SPLIT  = 0.1 # % of Train Data splitted into Validation Data
MODEL_PATH = "best_model.npy"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=15)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=128)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy", choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0005)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier", choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment_1")

    return parser.parse_args()

def load_model(model_path):
    # Load trained model!
    data = np.load(model_path, allow_pickle=True).item()
    return data

def evaluate_model(model, X_test, y_test, loss):
    loss_fn, _ = get_loss(loss)
    logits = np.concatenate([model.forward(X_test[i: i + 256]) for i in range(0, len(X_test), 256)], axis=0)
    y_int = np.argmax(y_test, axis=1)
    preds = np.argmax(logits, axis=1)
    return {
        "logits":    logits,
        "loss":      float(loss_fn(logits, y_test)),
        "accuracy":  float(accuracy_score(y_int, preds)),
        "precision": float(precision_score(y_int, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_int, preds,    average="macro", zero_division=0)),
        "f1":        float(f1_score(y_int, preds,        average="macro", zero_division=0)),
    }

def main():
    args = parse_arguments()
    # Data
    print(f"[Data] Loading {args.dataset} ...")
    _, _, (X_test, y_test_oh), (_, _, y_test_int) = load_dataset(args.dataset, val_split=VAL_SPLIT)
    print(f"  testing {X_test.shape[0]} images")
    # Model
    model   = NeuralNetwork(args)
    weights = load_model(MODEL_PATH)
    model.set_weights(weights)
    print(f"[Model] Loaded weights from {MODEL_PATH}")
    # Evaluate
    results = evaluate_model(model, X_test, y_test_oh, args.loss)
    for k, v in results.items():
        if k != "logits":
            print(f"  {k.capitalize():12s}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_int, np.argmax(results["logits"], axis=1)))
    print("\nEvaluation Completed!")
    return results

if __name__ == "__main__":
    main()