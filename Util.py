"""
This module contains utility functions and classes for common operations used in the Machine Learning project.

Classes:
    Data: Represents input and output data.
    Ratio: Represents the split ratio for training, validation, and testing datasets.
    Dataset: Handles loading and splitting of datasets.

Functions:
    is_tensor(array: np.ndarray) -> bool: Checks if the array is a valid Tensor (n-dimensional array).
    is_matrix(array: np.ndarray) -> bool: Checks if the array is a valid Matrix (2D array).
    is_vector(array: np.ndarray) -> bool: Checks if the array is a valid Vector (1D array).
    activation(x: np.ndarray, derivative: bool, mode: str) -> np.ndarray: Applies an activation function to the input array.
"""

import numpy as np
import pandas as pd

# Define Tensor, Matrix, and Vector as np arrays with respective dimensions
Tensor = np.ndarray  # A general n-dimensional array
Matrix = np.ndarray  # A 2D array (n x m)
Vector = np.ndarray  # A 1D array (n,)


def is_tensor(array: np.ndarray) -> bool:
    """
    Check if the array is a valid Tensor (n-dimensional array).

    Args:
        array (np.ndarray): The array to check.

    Returns:
        bool: True if the array is a Tensor, False otherwise.
    """
    #TODO Validate that the input array is not None and is a valid numpy array
    #TODO Add error handling for invalid inputs
    return isinstance(array, np.ndarray)


def is_matrix(array: np.ndarray) -> bool:
    """
    Check if the array is a valid Matrix (2D array).

    Args:
        array (np.ndarray): The array to check.

    Returns:
        bool: True if the array is a Matrix, False otherwise.
    """
    #TODO Validate that the input array is not None and is a valid numpy array
    #TODO Add error handling for invalid inputs
    return isinstance(array, np.ndarray) and array.ndim == 2


def is_vector(array: np.ndarray) -> bool:
    """
    Check if the array is a valid Vector (1D array).

    Args:
        array (np.ndarray): The array to check.

    Returns:
        bool: True if the array is a Vector, False otherwise.
    """
    #TODO Validate that the input array is not None and is a valid numpy array
    #TODO Add error handling for invalid inputs
    return isinstance(array, np.ndarray) and array.ndim == 1


def activation(x: Vector, derivative: bool= False, mode: str = "sigmoid") -> Vector:
    """
    Apply an activation function (e.g., sigmoid) to the input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        Vector: The transformed array after applying the activation function.
    """
    #TODO assert args are coorectly passed

    match mode:
        case "sigmoid":
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid) if derivative else sigmoid
            
        case _:
            raise ValueError(f"activation mode '{mode}' not recognized.")

def cost(x: Vector, y: Vector, derivative: bool= False, mode: str= "MSE") -> Vector:
    #TODO assert args are correctly passed
    match mode:
        #TODO Mean Squared Error
        case "MSE":
            if derivative:
                return 2 * (x - y)
            else:
                return (x - y) ** 2
            


# Convert to numpy arrays
def to_data(df: pd.DataFrame,
            label_column: str = "label") -> "Data":
    features = df.drop(columns=[label_column]).to_numpy()
    labels = df[label_column].to_numpy()
    return Data(features, labels)

class Data:
    """
    Represents input and output data.

    Attributes:
        input (np.array): The input data.
        output (np.array): The output data.
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray):
        
        self.features: np.ndarray = features
        self.labels: np.ndarray = labels


class Ratio:
    """
    Represents the split ratio for training, validation, and testing datasets.

    Attributes:
        training (float): The ratio of training data.
        validation (float): The ratio of validation data.
        testing (float): The ratio of testing data.
    """

    def __init__(self,
                 training: float,
                 validation: float,
                 testing: float):
        
        self.training: float = training
        self.validation: float = validation
        self.testing: float = testing

        # Check if the sum of ratios equals 1
        if not np.isclose(self.training + self.validation + self.testing, 1.0):
            raise ValueError("Sum of ratios must equal 1.")

class Dataset:
    """
    Handles loading and splitting of datasets.

    Attributes:
        path (str): The path to the dataset file.
        split_ratio (Ratio): The split ratio for the dataset.
        training (Data): The training data.
        validation (Data): The validation data.
        testing (Data): The testing data.

    Methods:
        load(): Loads the dataset from the specified path.
        split(): Splits the dataset into training, validation, and testing sets.
    """
    def __init__(self,
                 path: str,
                 split_ratio: Ratio):
        
        self.path: str = path#TODO assert path is correct 
        self.split_ratio: Ratio = split_ratio
        self.batches: list[Data] = []
        self.data: pd.DataFrame
           

        self.training: Data
        self.validation: Data
        self.testing: Data
    
    #TODO
    def load(self):
        if self.path.endswith(".csv"):
            self.data = pd.read_csv(self.path)
        elif self.path.endswith(".xlsx"):
            self.data = pd.read_excel(self.path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    #TODO
    def split(self, 
            shuffle: bool = True,
            label_column: str = "label"):
        """
        Splits the loaded dataset into training, validation, and testing sets.
        
        Args:
            label_column (str): Name of the column to be treated as the label.
        """
        if not hasattr(self, "data"):
            raise RuntimeError("You must call `load()` before `split()`.")

        df = self.data

        # Shuffle the DataFrame
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Compute split indices
        total_samples = len(df)
        train_end = int(total_samples * self.split_ratio.training)
        val_end = train_end + int(total_samples * self.split_ratio.validation)

        # Split DataFrame
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        self.training = to_data(train_df, label_column)
        self.validation = to_data(val_df, label_column)
        self.testing = to_data(test_df, label_column)


    def split_batches(self, batch_size: int, shuffle: bool = True) -> list[Data]:
        """
        Splits the training dataset into mini-batches.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data before splitting.

        Returns:
            list[Data]: List of mini-batch Data objects.
        """


        features = self.training.features
        labels = self.training.labels
        num_samples = features.shape[0]

        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            features = features[indices]
            labels = labels[indices]

        self.batches = [
            Data(
                features=features[i:i+batch_size],
                labels=labels[i:i+batch_size]
            )
            for i in range(0, num_samples, batch_size)
        ]
        return self.batches