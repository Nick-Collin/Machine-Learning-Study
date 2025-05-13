import numpy as np
import logging 

Tensor = np.ndarray
Matrix = np.ndarray
Vector = np.ndarray

def is_tensor(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray)

def is_matrix(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 2

def is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1

def activation(x: Vector, derivative: bool= False, mode: str = "sigmoid") -> Vector:
    match mode:
        case "sigmoid":
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid) if derivative else sigmoid
            
        case _:
            raise ValueError(f"activation mode '{mode}' not recognized.")

def cost(x: Vector, y: Vector, derivative: bool= False, mode: str= "MSE") -> Vector:
    match mode:
        case "MSE":
            if derivative:
                return 2 * (x - y)
            else:
                return (x - y) ** 2



