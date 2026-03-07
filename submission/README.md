# DA6401 – Assignment 1: Multi-Layer Perceptron for Image Classification

A fully-featured, NumPy-only MLP implementation for MNIST and Fashion-MNIST classification.

---

## Links

| Resource | URL |
|---|---|
| **W&B Report** | _Paste your W&B public report link here_ |
| **GitHub Repo** | _Paste your public GitHub repository link here_ |

---

## Project Structure

```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py        # sigmoid, tanh, relu + derivatives
│   │   ├── layer.py              # DenseLayer (forward + backward, grad_W, grad_b)
│   │   ├── loss.py               # CrossEntropy, MSE (+ gradients)
│   │   ├── neural_network.py     # NeuralNetwork (get/set_weights)
│   │   └── optimizers.py         # SGD, Momentum, NAG, RMSProp
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py         # load_dataset, get_batches
│   │   └── metrics.py            # accuracy, precision, recall, F1, confusion_matrix
│   ├── train.py                  # training script (argparse CLI)
│   ├── inference.py              # inference script (argparse CLI)
│   ├── best_model.npy            # serialised best weights
│   └── best_config.json          # best hyperparameter configuration
├── wandb_experiments.py          # W&B report experiment runner
├── gradient_check.py             # numerical gradient verification
├── sweep_config.yaml             # W&B sweep configuration (≥100 runs)
└── README.md
```

---

## Installation

```bash
pip install numpy scikit-learn keras wandb matplotlib
```

---

## Training

```bash
cd src
python train.py \
    -d fashion_mnist \
    -e 20 \
    -b 64 \
    -l cross_entropy \
    -o rmsprop \
    -lr 0.001 \
    -wd 0.0005 \
    -nhl 3 \
    -sz 128 \
    -a relu \
    -w_i xavier \
    -w_p <your_wandb_project>
```

### All CLI Arguments

| Flag | Long | Default | Description |
|---|---|---|---|
| `-d` | `--dataset` | `fashion_mnist` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | `20` | Number of training epochs |
| `-b` | `--batch_size` | `64` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop` |
| `-lr` | `--learning_rate` | `0.001` | Initial learning rate |
| `-wd` | `--weight_decay` | `0.0005` | L2 regularisation coefficient |
| `-nhl` | `--num_layers` | `3` | Number of hidden layers |
| `-sz` | `--hidden_size` | `128` | Neurons per hidden layer (list supported) |
| `-a` | `--activation` | `relu` | `sigmoid`, `tanh`, `relu` |
| `-w_i` | `--weight_init` | `xavier` | `random` or `xavier` |
| `-w_p` | `--wandb_project` | `da6401_assignment_1` | W&B project name |

---

## Inference

```bash
cd src
python inference.py \
    -d fashion_mnist \
    -nhl 3 \
    -sz 128 \
    -a relu \
    -w_i xavier \
    --model_path src/best_model.npy
```

Outputs: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**.

---

## Hyperparameter Sweep (W&B)

```bash
# Register sweep
wandb sweep sweep_config.yaml

# Launch agents (run until ≥100 configs completed)
wandb agent <sweep_id> --count 100
```

---

## W&B Experiments (Report Sections)

```bash
# Run all report sections
python wandb_experiments.py --section all --project <your_project>

# Run individual section (q1, q3, q4, q5, q6, q8, q9, q10)
python wandb_experiments.py --section q3 --project <your_project>
```

---

## Gradient Check

Verifies the analytical gradients against numerical (centred finite-difference) gradients.

```bash
python gradient_check.py
```

---

## Implementation Highlights

### Architecture
- **Input**: 784-dim flattened image
- **Hidden layers**: 1–6 `DenseLayer` objects (sigmoid / tanh / ReLU)
- **Output**: Linear layer → raw logits (no softmax inside network)

### Key Design Choices
- Softmax is applied **only inside loss functions** (not in the network output)
- `DenseLayer.backward()` stores `self.grad_W` and `self.grad_b` after every call
- `NeuralNetwork.backward()` returns gradients **from last layer to first**
- `get_weights()` / `set_weights()` use a dict keyed by layer index for serialisation
- NAG uses a lookahead/undo pattern around each forward-backward step

### Optimisers
| Optimiser | Key Parameters |
|---|---|
| SGD | `lr` |
| Momentum | `lr`, `β = 0.9` |
| NAG | `lr`, `β = 0.9` |
| RMSProp | `lr`, `ρ = 0.9`, `ε = 1e-8` |

---

## Academic Integrity

This implementation was written entirely by the student.  
AI tools were used only as conceptual aids, not for code generation.
