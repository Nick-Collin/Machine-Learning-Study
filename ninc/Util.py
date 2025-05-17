import numpy as np
from ninc.DataHandeling import Dataset
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

def activation(x: Vector, derivative: bool = False, mode: str = "sigmoid") -> Vector:
    match mode:
        case "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig) if derivative else sig

        case "tanh":
            tanh_x = np.tanh(x)
            return 1 - tanh_x ** 2 if derivative else tanh_x

        case "relu":
            if derivative:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)

        case "leaky_relu":
            alpha = 0.01
            if derivative:
                return np.where(x > 0, 1, alpha)
            return np.where(x > 0, x, alpha * x)

        case "elu":
            alpha = 1.0
            if derivative:
                return np.where(x > 0, 1, alpha * np.exp(x))
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))

        case "softplus":
            if derivative:
                return 1 / (1 + np.exp(-x))  # derivative is sigmoid
            return np.log1p(np.exp(x))  # log(1 + e^x)

        case "softsign":
            if derivative:
                return 1 / (1 + np.abs(x)) ** 2
            return x / (1 + np.abs(x))
        case "linear":
            if derivative: return 1
            else: return x
        case _:
            raise ValueError(f"No activation function named: {mode}. Use [sigmoid, tanh, relu, leaky_relu, softplus, softsign, linear]")
        
def cost(x: Vector, y: Vector, derivative: bool= False, mode: str= "MSE") -> Vector:
    match mode:
        case "MSE":
            if derivative:
                return 2 * (x - y)
            else:
                return (x - y) ** 2
            
def norm_input(x: Vector, dta: Dataset) -> Vector:
    match dta.norm_mode:
        case "standardization":
            return (x - dta.feature_mean) / dta.feature_std
        
        case _:
            raise AttributeError(f"Couldnt resolve norm mode")

def denorm_output(y: Vector, dta: Dataset) -> Vector:
    assert hasattr(dta, 'label_mean') and hasattr(dta, 'label_std'), "Labels where not normalized! You shoudnt denormalize outputs"
    
    match dta.norm_mode:
        case "standardization":
            return y * dta.label_std + dta.label_mean

