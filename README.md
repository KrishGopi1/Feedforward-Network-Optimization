# Feedforward Network Optimization

A feedforward neural network built from scratch using NumPy and trained on the MNIST dataset.

The main goal of this project is to understand what happens inside a neural network during training instead of relying on a deep learning framework. The implementation includes forward propagation, backpropagation, mini-batch training, dropout, and the Adam optimizer.

## What is implemented

* Fully connected feedforward neural network
* ReLU activation for hidden layers
* Softmax activation for the output layer
* Cross-entropy loss
* He initialization
* Mini-batch gradient descent
* Dropout
* Adam optimizer
* Forward and backward propagation
* MNIST training and evaluation
* Accuracy calculation

The current network used for MNIST is:

```text
784 → 128 → 64 → 10
```

## Project Structure

```text
Feedforward-Network-Optimization/
│
├── src/
│   ├── activation.py
│   ├── layers.py
│   ├── loss.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── logs/
├── data/
├── requirements.txt
└── README.md
```

### `layers.py`

Contains parameter initialization. The network uses He initialization for the weights of the layers.

### `model.py`

Contains most of the neural network implementation:

* Mini-batch creation
* Forward propagation
* Dropout
* Backpropagation
* Standard gradient descent
* Adam parameter updates
* Prediction
* Accuracy calculation

### `train.py`

Handles the training loop. It creates mini-batches, runs forward and backward propagation, calculates the loss, and updates the parameters using Adam.

### `main.py`

Loads MNIST, creates the network, trains it, and prints the training and test accuracy.

## Training

The network is trained using mini-batches with Adam.

Dropout is applied to the hidden layers during training with a keep probability of `0.8`.

The training setup currently uses:

```text
Learning rate: 0.001
Mini-batch size: 64
Epochs: 100
Architecture: 784 → 128 → 64 → 10
Optimizer: Adam
```

## Running the project

Clone the repository:

```bash
git clone https://github.com/KrishGopi1/Feedforward-Network-Optimization.git
cd Feedforward-Network-Optimization
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/main.py
```

The program prints the training cost during training and the final training and test accuracy.

## Why build it from scratch?

Frameworks such as PyTorch and TensorFlow make neural network development much easier, but they also hide many of the operations happening underneath.

This project is mainly about implementing those operations directly:

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Dropout
  ↓
Linear Layer
  ↓
ReLU
  ↓
Dropout
  ↓
Softmax
  ↓
Prediction
```

During backpropagation, the gradients are calculated manually and passed to the Adam optimizer to update the weights and biases.

## What I am experimenting with

The project is being used to understand how different choices affect neural network training, particularly:

* Weight initialization
* Learning rate
* Mini-batch size
* Dropout
* Optimization algorithms
* Network architecture
* Training stability and convergence

The next step is to extend the same from-scratch approach to convolutional networks, starting with LeNet-5.

## Requirements

* Python 3
* NumPy
* Matplotlib
* MNIST dataset

See `requirements.txt` for the Python dependencies.

## Notes

This is primarily a learning and experimentation project. The focus is on understanding the implementation of neural network components rather than achieving state-of-the-art MNIST accuracy.
