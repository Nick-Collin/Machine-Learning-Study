# Ninc: A Simple Neural Network Framework

Welcome to **Ninc**, a lightweight neural network framework designed for educational purposes. This project is a practical implementation of machine learning concepts, created to help you learn and experiment with neural networks. Enjoy exploring and contributing!

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Optimizers](#optimizers)
- [Model Persistence](#model-persistence)
- [Extending Ninc](#extending-ninc)
  - [Custom Activation Functions](#custom-activation-functions)
  - [Custom Cost Functions](#custom-cost-functions)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [Contact](#contact)

---

## Features

- Lightweight and easy-to-understand neural network framework.
- Supports dataset handling, including loading, splitting, batching, and normalization.
- Customizable neural network architecture with various activation functions and initialization methods.
- Multiple optimization algorithms, including SGD, Adam, and RMSProp.
- Comprehensive training and validation tools.

---

## Requirements

- Python 3.8 or higher
- Dependencies: `numpy`, `pandas`

---

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nick-Collin/Machine-Learning-Study.git
   cd Machine-Learning-Study
   ```
2. Install the package and its dependencies:
   ```bash
   pip install .
   ```

### You can also run the example scripts:

```bash
python examples/iris.py
```

---

## Usage

### Step-by-Step Guide

1. **Prepare Your Dataset**

   ```python
   from ninc.DataHandling import Dataset, Ratio

   dataset_path = "Data/LinearRegression.csv"
   split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

   dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
   dataset.load()
   dataset.split(label_column="out")
   dataset.normalize(normalize_labels=True)
   ```
2. **Define Neural Network Layers**

   ```python
   from ninc.Neural_Network import Layer

   layers = [
       Layer(num_of_units=1, activation_mode="linear", initialization_mode="He")
   ]
   ```
3. **Initialize the Neural Network**

   ```python
   from ninc.Neural_Network import NeuralNetwork

   nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)
   ```
4. **Set Up a Trainer**

   ```python
   from ninc.Trainers import Trainer, PolyakMomentum

   optimizer = PolyakMomentum()
   trainer = Trainer(nn=nn, optimizer=optimizer, epochs=1000, learning_rate=0.01, checkpoint=100)
   ```
5. **Train the Model**

   ```python
   trainer.train(batched=True, batch_size=32, shuffle=True)
   ```
6. **Evaluate the Model**
7. **Full Example**

   ```python
   from ninc.Util import logging, np, norm_input, denorm_output
   from ninc.Neural_Network import NeuralNetwork, Layer
   from ninc.Trainers import Trainer, PolyakMomentum
   from ninc.DataHandling import Dataset, Ratio

   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   def main():
       # Define dataset path and split ratio
       dataset_path = "Data/LinearRegression.csv"
       split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

       # Load and split dataset
       dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
       dataset.load()
       dataset.split(label_column="out")
       dataset.normalize(normalize_labels= True)

       # Define neural network architecture
       layers = [ # Match input features (2)
           Layer(num_of_units=1, activation_mode="linear", initialization_mode= "He")
       ]
       nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)

       # Define optimizer and trainer
       optimizer = PolyakMomentum()
       trainer = Trainer(nn=nn, optimizer=optimizer, epochs=1000, learning_rate=0.01, checkpoint=1)

       # Train the model
       trainer.train(batched=True, batch_size=32, shuffle=True)

       testing_mode(nn, dataset)

   def testing_mode(nn: NeuralNetwork, dataset):
       while True:
           try:
               x1 = int(input("x1: "))
               x2 = int(input("x2: "))
               inputs = np.array([x1, x2])

           except:
               print("Couldn't resolve inputs, bailing out!")
               exit()

           out = nn.propagate_forward(norm_input(inputs, dataset))
           out = denorm_output(out, dataset)
           print(f"Out: {out}")

   if __name__ == "__main__":
       main()
   ```

### Usage as a Library

After installation, you can import ninc in your Python scripts:

```python
from ninc.DataHandling import Dataset, Ratio
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, PolyakMomentum
```

---

## Optimizers

Ninc provides several optimization algorithms to train your neural networks efficiently. Each optimizer can be configured with specific parameters to suit your needs. Below are the available optimizers and their configuration options:

### SGD (Stochastic Gradient Descent)

- **Description:** The simplest optimizer. Updates weights using the gradient of the loss function.
- **Config:**
  - No additional parameters.
- **Usage:**
  ```python
  from ninc.Trainers import SGD
  optimizer = SGD()
  ```

### PolyakMomentum

- **Description:** Adds a momentum term to SGD, which helps accelerate convergence and reduce oscillations.
- **Config:**
  - `momentum` (float, default=0.9): The momentum factor (usually between 0 and 1).
- **Usage:**
  ```python
  from ninc.Trainers import PolyakMomentum
  optimizer = PolyakMomentum(momentum=0.9)
  ```

### RMSProp

- **Description:** Adapts the learning rate for each parameter using a moving average of squared gradients. Useful for non-stationary objectives.
- **Config:**
  - `beta` (float, default=0.9): Decay rate for the moving average.
  - `epsilon` (float, default=1e-8): Small value to avoid division by zero.
- **Usage:**
  ```python
  from ninc.Trainers import RMSProp
  optimizer = RMSProp(beta=0.9, epsilon=1e-8)
  ```

### Adam (Adaptive Moment Estimation)

- **Description:** Combines momentum and adaptive learning rates for robust performance. Maintains running averages of both the gradients and their squares.
- **Config:**
  - `beta1` (float, default=0.9): Decay rate for the first moment estimate (mean of gradients).
  - `beta2` (float, default=0.999): Decay rate for the second moment estimate (uncentered variance of gradients).
  - `epsilon` (float, default=1e-8): Small value to avoid division by zero.
- **Usage:**
  ```python
  from ninc.Trainers import Adam
  optimizer = Adam(beta1=0.9, beta2=0.999, epsilon=1e-8)
  ```

---

## Model Persistence

You can save and load your trained neural network models using the built-in `save` and `load` methods of the `NeuralNetwork` class. This allows you to persist model weights and biases to disk and reload them later for inference or further training.

### Saving a Model

```python
# After training
nn.save('model_weights.npz')
```

### Loading a Model

```python
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.DataHandling import Dataset

# Define your dataset and layers as before
# ...

# Load model
nn_loaded = NeuralNetwork.load('model_weights.npz', dataset=dataset, layers=layers, loss_func='MSE')
```

- The `save` method stores all layer weights and biases in a `.npz` file.
- The `load` method restores them into a new `NeuralNetwork` instance (you must provide the same architecture and dataset).

---

## Extending Ninc

Ninc is designed to be extensible. You can add your own activation and cost functions easily.

### Custom Activation Functions

You can easily extend Ninc with your own activation functions by subclassing `ActivationFunction` and registering your new class. This allows you to use your custom activation in any layer, just like the built-in ones.

**Example:**

```python
from ninc.Util import ActivationFunction, register_activation, activation
import numpy as np

class Swish(ActivationFunction):
    def __call__(self, x):
        return x / (1 + np.exp(-x))
    def derivative(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

# Register your custom activation
register_activation('swish', Swish())

# Use by name
result = activation(x, mode='swish')
# Or use directly
result = activation(x, mode=Swish())
```

You can now specify your custom activation by name in any `Layer`, or pass an instance directly to the `activation` function.

### Custom Cost Functions

You can extend Ninc with your own cost functions by subclassing `CostFunction` and registering your new class. This allows you to use your custom cost in training, just like the built-in ones.

**Example:**

```python
from ninc.Util import CostFunction, register_cost, cost
import numpy as np

class MAE(CostFunction):
    def __call__(self, x, y):
        return np.abs(x - y)
    def derivative(self, x, y):
        return np.where(x > y, 1, -1)

# Register your custom cost function
register_cost('MAE', MAE())

# Use by name
result = cost(x, y, mode='MAE')
# Or use directly
result = cost(x, y, mode=MAE())
```

You can now specify your custom cost by name in the `NeuralNetwork` constructor, or pass an instance directly to the `cost` function.

---

## FAQ

**Q: What data formats are supported?**
A: CSV and Excel files are supported for dataset loading.

**Q: Can I use custom activation or loss functions?**
A: Yes, you can specify the activation function in the `Layer` and the loss function in the `NeuralNetwork` constructor.

**Q: How do I handle missing data?**
A: Use the `handle_missing_data()` method in the `Dataset` class to fill or drop missing values.

**Q: Where can I find example scripts?**
A: See the `examples/` directory for ready-to-run scripts on different datasets.

---

## Troubleshooting

**Common Issues:**

- If you get ModuleNotFoundError, make sure you are running scripts from the project root or have installed the package correctly.
- For file not found errors, check that your dataset paths are correct and files exist in the Data/ directory.
- If you encounter issues with dependencies, ensure you have the correct Python version and have installed all required packages.
- For further help, open an issue on GitHub or contact the maintainer.

---

## Contributors

This project is open for contributions! Currently maintained by **Nicolas Pinho**. Feel free to fork, improve, and submit pull requests.

---

## Contact

For questions or support, please contact **Nicolas Pinho** at [nicolas.pinho.rj@gmail.com].
