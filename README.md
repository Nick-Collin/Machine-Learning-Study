# Ninc, A Simple Neural Network Framework

This is a project I've been working on lately for educational purposes. I've been studying Machine Learning on my own for a while, and I've decided to put it into practice. Well, that's the outcome. Hope you like it!

## Components

### DataHandeling.py

#### Description

This module is responsible for handling datasets, including loading, splitting, batching, and normalization. It provides utilities to preprocess data for training and testing.

#### Classes

- **Data**: Represents a dataset with features and labels.
- **Ratio**: Defines the split ratio for training, validation, and testing datasets.
- **Dataset**: Handles dataset loading, splitting, batching, and normalization.

#### Functions

- **to_data**: Converts a pandas DataFrame into a `Data` object.

### NeuralNetwork.py

#### Description

This module defines the structure and behavior of neural networks, including layers, forward propagation, and backward propagation.

#### Classes

- **NeuralNetwork**: Represents a neural network with multiple layers and a loss function.
- **Layer**: Represents a single layer in the neural network, including weight initialization, forward pass, and gradient computation.

### Trainers.py

#### Description

This module provides tools for training neural networks using various optimization algorithms.

#### Classes

- **Optimizer**: Base class for optimization algorithms.
- **SGD**: Implements Stochastic Gradient Descent.
- **PolyakMomentum**: Implements momentum-based optimization.
- **RMSProp**: Implements RMSProp optimization.
- **Adam**: Implements the Adam optimization algorithm.
- **Trainer**: Manages the training process, including batching, validation, and weight updates.

### Util.py

#### Description

This module provides utility functions and types for neural network operations, including activation functions, cost functions, and normalization utilities.

#### Functions

- **activation**: Implements various activation functions and their derivatives.
- **cost**: Implements cost functions and their derivatives.
- **norm_input**: Normalizes input features based on the dataset's normalization parameters.
- **denorm_output**: Denormalizes output predictions based on the dataset's normalization parameters.

## Installation

To use this framework, clone the repository and ensure you have Python installed along with the required libraries, such as `numpy` and `pandas`.

## Usage

1. Prepare your dataset in the `Data` folder (e.g., `LinearRegression.csv`).
2. Configure your dataset path and split ratio.

   ```python
   from ninc.DataHadeling import Dataset, Ratio

   dataset_path = "Data/LinearRegression.csv"
   split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

   dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
   ```
3. Load and split your dataset.

   ```python
   dataset.load()
   # Label_column gets the name of the column
   dataset.split(label_column= "out") 
   ```
4. Normalize(Optional)

   ```python
   # If normalize_labels is set to true the outputs will be normalized as well
   dataset.normalize(normalize_labels= True)
   ```
5. Configure the Layers

    ```python
    from ninc.NeuralNetwork import Layer

    layers =[
        Layer(num_of_units=1, activation_mode="linear", initialization_mode= "He")
    ]
    ```
6. Configure the Neural Network

    ```python
    from ninc.NeuralNetwork import NeuralNetwork

    nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)
    ```

## Contributors

This project is open for contributors. Currently, it's only me, Nicolas Pinho. Please feel free to contribute!
