# DA6401 – Assignment 1: Multi-Layer Perceptron for Image Classification

## Links

| Resource | URL |
|---|---|
| **W&B Report** | https://api.wandb.ai/links/me25m079-indian-institute-of-technology-madras/duoncekt |
| **GitHub Repo** |https://github.com/me25m079/DA6401_Assignment_1_2026.git |

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
