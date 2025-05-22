from ninc.Util import Vector
from abc import ABC, abstractmethod
import numpy as np
import logging

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: Vector) -> Vector:
        pass

    @abstractmethod
    def derivative(self, x: Vector) -> Vector:
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: Vector) -> Vector:
        sig = self.__call__(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return np.tanh(x)

    def derivative(self, x: Vector) -> Vector:
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2


class ReLU(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return np.maximum(0, x)

    def derivative(self, x: Vector) -> Vector:
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x: Vector) -> Vector:
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: Vector) -> Vector:
        return np.where(x > 0, 1, self.alpha)


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x: Vector) -> Vector:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: Vector) -> Vector:
        return np.where(x > 0, 1, self.alpha * np.exp(x))


class Softplus(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return np.log1p(np.exp(x))

    def derivative(self, x: Vector) -> Vector:
        return 1 / (1 + np.exp(-x))


class Softsign(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return x / (1 + np.abs(x))

    def derivative(self, x: Vector) -> Vector:
        return 1 / (1 + np.abs(x)) ** 2


class Linear(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        return x

    def derivative(self, x: Vector) -> Vector:
        return np.ones_like(x)


class Softmax(ActivationFunction):
    def __call__(self, x: Vector) -> Vector:
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x: Vector) -> Vector:
        # Returns the Jacobian matrix for each input vector
        s = self.__call__(x)
        # For a vector input, return the Jacobian matrix
        # For batch input, return array of Jacobians
        if s.ndim == 1:
            return np.diagflat(s) - np.outer(s, s)
        else:
            # For each sample in the batch, compute Jacobian
            return np.array([np.diagflat(si) - np.outer(si, si) for si in s])


# Registry for built-in and custom activations
_activation_registry = {
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'relu': ReLU(),
    'leaky_relu': LeakyReLU(),
    'elu': ELU(),
    'softplus': Softplus(),
    'softsign': Softsign(),
    'linear': Linear(),
    'softmax': Softmax(),
}

def register_activation(name: str, 
                        activation: ActivationFunction):
    
    _activation_registry[name] = activation

def get_activation(mode: str | ActivationFunction) -> ActivationFunction:
    if isinstance(mode, ActivationFunction):
        return mode
    
    if mode in _activation_registry:
        return _activation_registry[mode]
    
    raise ValueError(f"No activation function named: {mode}. Use one of {list(_activation_registry.keys())} or register your own.")

def activation(x: Vector, 
               derivative: bool = False, 
               mode: str | ActivationFunction = "sigmoid") -> Vector:
    try:
        act = get_activation(mode)
        logging.debug(f"Activation function '{act.__class__.__name__}' called with input shape {x.shape}")

        if derivative:
            return act.derivative(x)
        
        else:
            return act(x)
        
    except Exception as e:
        logging.error(f"Error in activation function '{mode}': {e}")
        raise 