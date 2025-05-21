# Ninc: A Simple Neural Network Framework

Welcome to **Ninc**, a lightweight neural network framework designed for educational purposes. This project is a practical implementation of machine learning concepts, created to help you learn and experiment with neural networks. Enjoy exploring and contributing!

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Logging](#logging)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [You can also run the example scripts](#you-can-also-run-the-example-scripts)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Step-by-Step Guide](#step-by-step-guide)
    1. [Prepare Your Dataset](#prepare-your-dataset)
    2. [Define Neural Network Layers](#define-neural-network-layers)
    3. [Initialize the Neural Network](#initialize-the-neural-network)
    4. [Set Up a Trainer](#set-up-a-trainer)
    5. [Train the Model](#train-the-model)
    6. [Evaluate the Model](#evaluate-the-model)
    7. [Make Predictions](#make-predictions)
    8. [Save and Load Your Model](#save-and-load-your-model)
- [Optimizers](#optimizers)
  - [SGD (Stochastic Gradient Descent)](#sgd-stochastic-gradient-descent)
  - [PolyakMomentum](#polyakmomentum)
  - [RMSProp](#rmsprop)
  - [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)
- [Model Persistence](#model-persistence)
  - [Saving a Model](#saving-a-model)
  - [Loading a Model](#loading-a-model)
- [Mathematical Foundations](#mathematical-foundations)
  - [Forward Propagation](#forward-propagation)
  - [Backpropagation](#backpropagation)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers-1)
- [Extending Ninc](#extending-ninc)
  - [Custom Activation Functions](#custom-activation-functions)
  - [Custom Cost Functions](#custom-cost-functions)
  - [Custom Optimizers](#custom-optimizers)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [Contact](#contact)
- [Project Structure](#project-structure)
  - [ninc/DataHandling.py](#nincdatahandlingpy)
    - [Functions](#functions)
    - [Classes](#classes)
  - [ninc/Neural_Network.py](#nincneural_networkpy)
    - [Classes](#classes-1)
  - [ninc/Trainers.py](#ninctrainerspy)
    - [Classes](#classes-2)
  - [ninc/Util.py](#nincutilpy)
    - [Type Aliases](#type-aliases)
    - [Utility Functions](#utility-functions)
    - [Activation Functions](#activation-functions)
    - [Cost Functions](#cost-functions)
    - [Normalization Helpers](#normalization-helpers)

---

## Features

- Supports dataset handling, including loading, splitting, batching, and normalization.
- Customizable neural network architecture with various activation functions and initialization methods.
- Multiple optimization algorithms, including SGD, Adam, and RMSProp.
- Comprehensive training and validation tools.
- Built-in cost functions: MSE (Mean Squared Error), CrossEntropy (Categorical/Binary Cross Entropy).

---

## Requirements

- Python 3.8 or higher
- Dependencies: `numpy`, `pandas`

---

## Logging

Ninc uses Python's built-in `logging` module to provide detailed information about the internal operations of the framework. Logging is helpful for debugging, monitoring training progress, and understanding data processing steps.

> **Note:** Some example scripts redirect `sys.stdout` and `sys.stderr` to a file for logging all output. For most use cases, it is recommended to use the logging module's file handler as shown below for more control and flexibility.

### Enabling Logging

To see log messages, configure the logging level and format at the start of your script. For example:

```python
import logging

# Show INFO and above messages (INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)

# For more detailed output (including DEBUG messages):
# logging.basicConfig(level=logging.DEBUG)
```

You can also customize the log format and output file:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename='ninc.log',  # Log to a file instead of the console
)
```

### Example

```python
import logging
logging.basicConfig(level=logging.INFO)

from ninc.DataHandling import Dataset, Ratio
# ... rest of your code ...
```

Now, as you run your training or data processing, you will see log messages such as:

```
INFO:root:Loading data from Data/iris.csv
INFO:root:Splitting data into training, validation, and testing sets with ratios <ninc.DataHandling.Ratio object at 0x...>
INFO:root:Normalizing data using standardization mode
INFO:root:Epoch 0/100 completed. Processed 32 samples.
INFO:root:Validation loss: 0.1234
```

Set the level to `DEBUG` for even more detailed output, including internal computations and shapes.

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

### Quick Start

Here's a minimal example to get you started with Ninc:

```python
import logging
logging.basicConfig(level=logging.INFO)

from ninc.DataHandling import Dataset, Ratio
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, Adam

# Load and prepare data
dataset = Dataset(path="Data/iris.csv", split_ratio=Ratio(0.7, 0.2, 0.1))
dataset.load()
dataset.split(label_column="species", one_hot=True)
dataset.normalize()

# Define network architecture
layers = [
    Layer(num_of_units=8, activation_mode="relu", initialization_mode="He"),
    Layer(num_of_units=3, activation_mode="softsign", initialization_mode="Xavier")
]

# Initialize network and trainer
nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)
optimizer = Adam()
trainer = Trainer(nn=nn, optimizer=optimizer, epochs=20, learning_rate=0.01)

# Train and evaluate
trainer.train()
trainer.test()
```

This example loads the Iris dataset, builds a simple neural network, trains it, and evaluates its performance. For more details, see the step-by-step guide below.

### Step-by-Step Guide

#### 1. Prepare Your Dataset

```python
from ninc.DataHandling import Dataset, Ratio

dataset_path = "Data/iris.csv"
split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
dataset.load()
dataset.split(label_column="species", one_hot=True)  # Use your label column name
dataset.normalize(normalize_labels=True)
```

- `label_column`: Name of the column with your target variable.
- `one_hot=True`: Use for classification tasks with categorical labels.

#### 2. Define Neural Network Layers

```python
from ninc.Neural_Network import Layer

layers = [
    Layer(num_of_units=8, activation_mode="relu", initialization_mode="He"),
    Layer(num_of_units=3, activation_mode="softsign", initialization_mode="Xavier")  # For 3 classes
]
```

- Adjust `num_of_units` and `activation_mode` for your problem.

#### 3. Initialize the Neural Network

```python
from ninc.Neural_Network import NeuralNetwork

nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)
```

- `loss_func`: Use `"MSE"` for regression or multi-class classification, `"CrossEntropy"` for binary or multi-class classification.

#### 4. Set Up a Trainer

```python
from ninc.Trainers import Trainer, Adam

optimizer = Adam(beta1=0.9, beta2=0.999, epsilon=1e-8)
trainer = Trainer(nn=nn, optimizer=optimizer, epochs=100, learning_rate=0.01, checkpoint=10)
```

- Choose optimizer: `SGD`, `PolyakMomentum`, `RMSProp`, or `Adam`.

#### 5. Train the Model

```python
trainer.train(batched=True, batch_size=32, shuffle=True)
```

- Use batching for faster and more stable training.

#### 6. Evaluate the Model

```python
trainer.test()
trainer.calculate_precision(dataset_type="testing")  # For classification
```

- `calculate_precision` is useful for classification tasks.

#### 7. Make Predictions

After training, you can use your model for inference:

```python
from ninc.Util import norm_input, denorm_output
import numpy as np

# Example input (replace with your own feature values)
inputs = np.array([5.1, 3.5, 1.4, 0.2])
inputs_norm = norm_input(inputs, dataset.feature_mean, dataset.feature_std, dataset.norm_mode)
output = nn.propagate_forward(inputs_norm)
output_denorm = denorm_output(output, dataset.label_mean, dataset.label_std, dataset.norm_mode)
print(output_denorm)
```

#### 8. Save and Load Your Model

```python
# Save
nn.save('model_weights.npz')

# Load
from ninc.Neural_Network import NeuralNetwork
nn_loaded = NeuralNetwork.load('model_weights.npz', dataset=dataset, layers=layers, loss_func='MSE')
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

## Mathematical Foundations

This section provides the mathematical background for the core algorithms implemented in Ninc, including forward propagation, backpropagation, loss functions, and optimizers. Mathematical notation uses LaTeX-style formatting for clarity.

### Forward Propagation

In a feedforward neural network, each layer computes its output as:

$$
\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})
$$

where

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

- $\mathbf{a}^{(l)}$: activations of layer $l$
- $\mathbf{W}^{(l)}$: weight matrix of layer $l$
- $\mathbf{b}^{(l)}$: bias vector of layer $l$
- $f^{(l)}$: activation function (e.g., sigmoid, relu)

### Backpropagation

Backpropagation computes gradients of the loss $\mathcal{L}$ with respect to weights and biases using the chain rule:

$$
\delta^{(l)} = (\mathbf{W}^{(l+1)T} \delta^{(l+1)}) \odot f'^{(l)}(\mathbf{z}^{(l)})
$$

where $\odot$ denotes element-wise multiplication, and $f'$ is the derivative of the activation function.

Gradients for parameter updates:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (\mathbf{a}^{(l-1)})^T
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

### Loss Functions

- **Mean Squared Error (MSE):**

  $$
  \mathrm{MSE}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  $$
- **Categorical Cross-Entropy:**

  $$
  \mathrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^K y_k \log(\hat{y}_k)
  $$

### Optimizers

- **SGD (Stochastic Gradient Descent):**

  $$
  \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
  $$

  where $\eta$ is the learning rate.
- **Polyak Momentum:**

  $$
  v \leftarrow \gamma v + \eta \nabla_\theta \mathcal{L} \\
  \theta \leftarrow \theta - v
  $$

  where $\gamma$ is the momentum factor.
- **RMSProp:**

  $$
  s \leftarrow \beta s + (1-\beta) (\nabla_\theta \mathcal{L})^2 \\
  \theta \leftarrow \theta - \frac{\eta}{\sqrt{s + \epsilon}} \nabla_\theta \mathcal{L}
  $$

  where $\beta$ is the decay rate, $\epsilon$ is a small constant.
- **Adam:**

  $$
  m \leftarrow \beta_1 m + (1-\beta_1) \nabla_\theta \mathcal{L} \\
  v \leftarrow \beta_2 v + (1-\beta_2) (\nabla_\theta \mathcal{L})^2 \\
  \hat{m} = \frac{m}{1-\beta_1^t}, \quad \hat{v} = \frac{v}{1-\beta_2^t} \\
  \theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m}
  $$

---

## Test Coverage

A `tests/` directory is included with unit tests for core components. To run all tests, use:

```bash
python -m unittest discover tests
```

---

## Extending Ninc

Ninc is designed to be extensible. You can add your own activation and cost functions easily.

### Custom Activation Functions

You can easily extend Ninc with your own activation functions by subclassing `ActivationFunction` and registering your new class. This allows you to use your custom activation in any layer, just like the built-in ones.

**Built-in:**

- `sigmoid`: Standard sigmoid function, outputs values in (0, 1).
- `tanh`: Hyperbolic tangent, outputs values in (-1, 1).
- `relu`: Rectified Linear Unit, outputs 0 for negative inputs and x for positive inputs.
- `leaky_relu`: Leaky ReLU, allows a small, non-zero gradient when the unit is not active.
- `elu`: Exponential Linear Unit, smooths the output for negative values.
- `softplus`: Smooth approximation to ReLU.
- `softsign`: Smooth approximation to the sign function.
- `linear`: Identity function, outputs the input as is.

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

### Custom Cost Functions

You can extend Ninc with your own cost functions by subclassing `CostFunction` and registering your new class. This allows you to use your custom cost in training, just like the built-in ones.

**Built-in:**

- `MSE`: Mean Squared Error (default for regression)
- `CrossEntropy`: Cross Entropy Error (for classification)

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

### Custom Optimizers

You can extend Ninc with your own optimizers by subclassing the `Optimizer` base class and implementing the `update` method. This allows you to use your custom optimizer in the training process, just like the built-in ones.

**Example:**

```python
from ninc.Trainers import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, my_param=0.1):
        self.my_param = my_param

    def update(self, layer, learning_rate):
        # Implement your custom update logic here
        # For example, a simple scaled SGD:
        layer.weights -= learning_rate * self.my_param * layer.grad_weights
        layer.bias -= learning_rate * self.my_param * layer.grad_bias

# Use your custom optimizer
optimizer = MyOptimizer(my_param=0.5)
```

You can now pass your custom optimizer to the `Trainer` just like any built-in optimizer.

## FAQ

**Q: What data formats are supported?**
A: CSV and Excel files are supported for dataset loading.

**Q: Can I use custom activation or loss functions?**
A: Yes, you can specify the activation function in the `Layer` and the loss function in the `NeuralNetwork` constructor. Built-in loss functions are `"MSE"` and `"CrossEntropy"`.

**Q: How do I handle missing data?**
A: Use the `handle_missing_data()` method in the `Dataset` class to fill or drop missing values.

**Q: What happens if I provide an unsupported file format or a missing label column?**
A: The framework will raise an error message if you provide an unsupported file format (only `.csv` and `.xlsx` are supported) or if the label column is missing during dataset splitting.

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

---

## Project Structure

### ninc/DataHandling.py

Responsible for dataset management.

#### Functions

- `to_data(df, label_column="label", one_hot=False)`: Converts a DataFrame into a Data object, extracting features and labels, with optional one-hot encoding.

#### Classes

- **Data**
  - `__init__(features, labels)`: Stores features and labels as numpy arrays.
- **Ratio**
  - `__init__(training, validation, testing)`: Stores split ratios for dataset partitioning. Validates that the sum is 1.0.
- **Dataset**
  - `__init__(path, split_ratio)`: Initializes dataset path, split ratio, and placeholders for data and splits.
  - `load()`: Loads data from CSV or Excel file into a DataFrame.
  - `split(shuffle=True, label_column="label", one_hot=False)`: Splits data into training, validation, and testing sets, with optional shuffling and one-hot encoding.
  - `split_batches(batch_size, shuffle=True)`: Splits training data into batches for mini-batch training.
  - `normalize(mode="standardization", normalize_labels=False)`: Normalizes features (and optionally labels) using standardization or min-max scaling.
  - `handle_missing_data(columns, strategy="mean", placeholder=0)`: Handles missing data in specified columns using mean, median, mode, or drop strategies.

---

### ninc/Neural_Network.py

Defines the neural network architecture and core logic.

#### Classes

- **NeuralNetwork**
  - `__init__(dataset, layers, loss_func="MSE")`: Initializes the neural network with layers, loss function, and dataset. Sets up layer connections and weight shapes.
  - `propagate_forward(inputs)`: Feeds inputs through all layers, returning the output.
  - `propagate_backward(expected)`: Performs backpropagation to compute gradients for all layers.
  - `save(filepath)`: Saves all layer weights and biases to a .npz file.
  - `@classmethod load(filepath, dataset, layers, loss_func="MSE")`: Loads weights and biases from a .npz file into a new NeuralNetwork instance.
- **Layer**
  - `__init__(num_of_units, activation_mode="sigmoid", weights=None, bias=None, initialization_mode="Xavier", clipping=True, clip_size=500)`: Initializes a layer with units, activation, weights, bias, and initialization mode.
  - `initialize_weights(mode="Xavier")`: Initializes weights and biases using Xavier, He, or their normal variants.
  - `forward_pass(inputs)`: Computes the output of the layer for given inputs.
  - `get_deltas(next)`: Computes deltas for backpropagation using the next layer.
  - `get_gradient()`: Computes the gradient of the loss with respect to the weights.

---

### ninc/Trainers.py

Implements training routines and optimization algorithms.

#### Classes

- **Optimizer (base class)**
  - `update(layer, learning_rate)`: Abstract method to update layer parameters.
- **SGD(Optimizer)**
  - `update(layer, learning_rate)`: Standard stochastic gradient descent update.
- **PolyakMomentum(Optimizer)**
  - `__init__(momentum=0.9)`: Sets the momentum factor.
  - `update(layer, learning_rate)`: Updates weights and biases using momentum.
- **RMSProp(Optimizer)**
  - `__init__(beta=0.9, epsilon=1e-8)`: Sets decay rate and epsilon.
  - `update(layer, learning_rate)`: Updates weights and biases using RMSProp algorithm.
- **Adam(Optimizer)**
  - `__init__(beta1=0.9, beta2=0.999, epsilon=1e-8)`: Sets Adam parameters.
  - `update(layer, learning_rate)`: Updates weights and biases using Adam algorithm.
- **Trainer**
  - `__init__(nn, optimizer, epochs, learning_rate, checkpoint=10)`: Sets up the training process.
  - `train(batched=True, batch_size=32, shuffle=True)`: Runs the training loop, optionally using mini-batches.
  - `validate()`: Evaluates the model on the validation set.
  - `train_step(features, labels)`: Performs a single training step (forward, backward, update) for a batch.
  - `test()`: Evaluates the model on the test set and logs the loss.
  - `calculate_precision(dataset_type="testing")`: Calculates classification precision on the specified dataset.

---

### ninc/Util.py

Utility functions and extensibility features.

#### Type Aliases

- `Tensor`, `Matrix`, `Vector`: Aliases for numpy arrays.

#### Utility Functions

- `is_tensor(array)`: Checks if input is a numpy ndarray.
- `is_matrix(array)`: Checks if input is a 2D numpy array.
- `is_vector(array)`: Checks if input is a 1D numpy array.

#### Activation Functions

- **ActivationFunction (abstract base class)**: Requires __call__ and derivative methods.
- **Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Softplus, Softsign, Linear**: Implementations of common activation functions.
- `register_activation(name, activation)`: Registers a custom activation function.
- `get_activation(mode)`: Retrieves an activation function by name or instance.
- `activation(x, derivative=False, mode="sigmoid")`: Applies the activation function or its derivative.

#### Cost Functions

- **CostFunction (abstract base class)**: Requires __call__ and derivative methods.
- **MSE, CrossEntropy**: Implementations of mean squared error and cross entropy error.
- `register_cost(name, cost)`: Registers a custom cost function.
- `get_cost(mode)`: Retrieves a cost function by name or instance.
- `cost(x, y, derivative=False, mode="MSE")`: Applies the cost function or its derivative.

#### Normalization Helpers

- `norm_input(x, feature_mean, feature_std, norm_mode)`: Normalizes input features.
- `denorm_output(y, label_mean, label_std, norm_mode)`: Denormalizes output predictions.
