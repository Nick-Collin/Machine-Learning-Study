# Ninc: A Simple Neural Network Framework

Welcome to **Ninc**, a lightweight neural network framework designed for educational purposes. This project is a practical implementation of machine learning concepts, created to help you learn and experiment with neural networks. Enjoy exploring and contributing!

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

## Project Structure

### Modules

#### 1. **DataHandeling.py**

- **Purpose**: Handles datasets (loading, splitting, batching, normalization).
- **Key Classes**:
  - `Data`: Represents a dataset.
    - **Attributes**:
      - `features`: Feature matrix (numpy array).
      - `labels`: Label vector (numpy array).
  - `Ratio`: Defines train/validation/test split ratios.
    - **Arguments**:
      - `training` (float): Proportion of training data.
      - `validation` (float): Proportion of validation data.
      - `testing` (float): Proportion of testing data.
  - `Dataset`: Manages dataset operations.
    - **Methods**:
      - `load()`: Loads data from a file (CSV or Excel).
      - `split(shuffle=True, label_column='label')`: Splits data into training, validation, and testing sets.
      - `split_batches(batch_size, shuffle=True)`: Splits training data into batches.
      - `normalize(mode='standardization', normalize_labels=False)`: Normalizes features and labels.
- **Key Function**:
  - `to_data(df, label_column='label')`: Converts a pandas DataFrame into a `Data` object.

#### 2. **Neural_Network.py**

- **Purpose**: Defines neural network architecture and operations.
- **Key Classes**:
  - `NeuralNetwork`: Represents a neural network.
    - **Arguments**:
      - `dataset` (Dataset): The dataset to train on.
      - `layers` (list[Layer]): List of layers in the network.
      - `loss_func` (str): Loss function to use (e.g., "MSE").
    - **Methods**:
      - `propagate_forward(inputs)`: Performs forward propagation.
      - `propagate_backward(expected)`: Performs backward propagation.
  - `Layer`: Represents a single layer in the network.
    - **Arguments**:
      - `num_of_units` (int): Number of neurons in the layer.
      - `activation_mode` (str): Activation function (e.g., "sigmoid").
      - `weights` (Matrix): Weight matrix (optional).
      - `bias` (Vector): Bias vector (optional).
      - `initialization_mode` (str): Weight initialization method (e.g., "Xavier").
    - **Methods**:
      - `initialize_weights(mode)`: Initializes weights and biases.
      - `forward_pass(inputs)`: Computes the output of the layer.
      - `get_deltas(next_layer)`: Computes deltas for backpropagation.
      - `get_gradient()`: Computes the gradient for weight updates.

#### 3. **Trainers.py**

- **Purpose**: Provides tools for training neural networks.
- **Key Classes**:
  - `Trainer`: Manages the training process.
    - **Arguments**:
      - `nn` (NeuralNetwork): The neural network to train.
      - `optimizer` (Optimizer): Optimization algorithm (e.g., SGD, Adam).
      - `epochs` (int): Number of training epochs.
      - `learning_rate` (float): Learning rate for optimization.
      - `checkpoint` (int): Interval for logging progress.
    - **Methods**:
      - `train(batched=True, batch_size=32, shuffle=True)`: Trains the network.
      - `validate()`: Evaluates the network on the validation set.
      - `train_step(features, labels)`: Performs a single training step.
  - Optimizers:
    - `SGD`: Stochastic Gradient Descent.
    - `PolyakMomentum`: Momentum-based optimizer.
    - `RMSProp`: Root Mean Square Propagation.
    - `Adam`: Adaptive Moment Estimation.

#### 4. **Util.py**

- **Purpose**: Utility functions for activation, cost functions, and normalization.
- **Key Functions**:
  - `activation(x, derivative=False, mode='sigmoid')`: Implements activation functions.
    - **Arguments**:
      - `x` (Vector): Input vector.
      - `derivative` (bool): Whether to compute the derivative.
      - `mode` (str): Activation function (e.g., "sigmoid", "relu").
    - **Returns**: Transformed vector.
  - `cost(x, y, derivative=False, mode='MSE')`: Computes cost functions.
    - **Arguments**:
      - `x` (Vector): Predicted values.
      - `y` (Vector): True values.
      - `derivative` (bool): Whether to compute the derivative.
      - `mode` (str): Cost function (e.g., "MSE").
    - **Returns**: Cost value or gradient.
  - `norm_input(x, dta)`: Normalizes input features.
    - **Arguments**:
      - `x` (Vector): Input vector.
      - `dta` (Dataset): Dataset object with normalization parameters.
    - **Returns**: Normalized vector.
  - `denorm_output(y, dta)`: Denormalizes output values.
    - **Arguments**:
      - `y` (Vector): Output vector.
      - `dta` (Dataset): Dataset object with normalization parameters.
    - **Returns**: Denormalized vector.

---

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas
   ```

---

## Usage

### Step-by-Step Guide

1. **Prepare Your Dataset**

   ```python
   from ninc.DataHandeling import Dataset, Ratio

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
   from ninc.DataHandeling import Dataset, Ratio

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

       # Evaluate the model (example)
       print("Training complete. Evaluate the model as needed.")

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

---

## Contributors

This project is open for contributions! Currently maintained by **Nicolas Pinho**. Feel free to fork, improve, and submit pull requests.

---

## Contact

For questions or support, please contact **Nicolas Pinho** at [nicolas.pinho.rj@gmail.com].
