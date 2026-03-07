import argparse
import json
import os
import sys
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(__file__)) # This is used to look in the src folder.

from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, get_batches

VAL_SPLIT     = 0.1
MODEL_SAVE_PATH = "best_model.npy"

def parse_arguments():
    #Command Line Interface arguments
    parser = argparse.ArgumentParser(description="Train a Neural Network!")
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=15)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=128)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy", choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0005)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3, help="Number of hidden layers")
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128], help="Number of neurons per hidden layer")
    parser.add_argument("-a",   "--activation",    type=str,   default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier", choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_assignment_1")
    # ----------------------------------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--no_wandb", action="store_true", default= False)
    return parser.parse_args()

def compute_metrics(y_true, y_pred):
    # Calculates Accuracy, Precision, Recall and F1-Score.
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,    average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred,        average="macro", zero_division=0)),
    }

def evaluate(model, X, y_oh, y_int, loss_fn, batch_size):
    logits = np.concatenate(
        [model.forward(X[s: s + batch_size]) for s in range(0, len(X), batch_size)],
        axis=0,
    )
    loss  = float(loss_fn(logits, y_oh))
    preds = np.argmax(logits, axis=1)
    return loss, compute_metrics(y_int, preds)

def main():
    # Main Training Function
    args = parse_arguments()
    # Data
    print(f"[Data] Loading {args.dataset} ...")
    (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh), (y_train_int, y_val_int, y_test_int) = load_dataset(args.dataset, val_split=VAL_SPLIT)
    print(f"  train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")
    # Model
    model     = NeuralNetwork(args)
    print(model)
    loss_fn, loss_grad_fn = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    print(f"[Loss Function] {loss_fn}")
    print(f"[Optimizer] {optimizer}")
    # wandb logging needed or not?
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    # ── Training loop ──────────────────────────────────────────────────────────
    best_f1, best_weights = -1.0, None
    is_nag = args.optimizer == "nag"
    for epoch in range(1, args.epochs + 1):
        tr_loss_sum, nb = 0.0, 0
        for Xb, yb in get_batches(X_train, y_train_oh, args.batch_size, shuffle=True):
            if is_nag:
                optimizer.apply_lookahead(model.layers)
            logits = model.forward(Xb)
            loss   = loss_fn(logits, yb)
            if is_nag:
                optimizer.undo_lookahead(model.layers)
            model.backward(yb, logits)
            optimizer.update(model.layers)
            tr_loss_sum += float(loss)
            nb          += 1
        val_loss, val_met = evaluate(model, X_val, y_val_oh, y_val_int, loss_fn, args.batch_size)
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss_sum/nb:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_met['accuracy']:.4f}  "
              f"val_f1={val_met['f1']:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": tr_loss_sum / nb,
                       "val_loss": val_loss,
                       **{f"val_{k}": v for k, v in val_met.items()},
                       **model.gradient_norms(), **model.activation_stats()})
        if val_met["f1"] > best_f1:
            best_f1, best_weights = val_met["f1"], model.get_weights()

    # Test Data Evaluation
    model.set_weights(best_weights)
    _, test_met = evaluate(model, X_test, y_test_oh, y_test_int, loss_fn, args.batch_size)
    print("\n")
    for k, v in test_met.items():
        print(f"  {k}: {v:.4f}")
    if use_wandb:
        wandb.log({f"test_{k}": v for k, v in test_met.items()})
        wandb.finish()
    # Saving the model
    np.save(MODEL_SAVE_PATH, best_weights)
    print(f"\nSaved model  → {MODEL_SAVE_PATH}")
    config_path = MODEL_SAVE_PATH.replace("_model.npy", "_config.json")
    with open(config_path, "w") as f:
        json.dump({**vars(args), "val_f1": best_f1,
                   **{f"test_{k}": v for k, v in test_met.items()}}, f, indent=2)

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
