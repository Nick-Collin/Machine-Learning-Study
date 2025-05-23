from neurolite.Util import Vector
from abc import ABC, abstractmethod
import numpy as np
import logging

'''
TODO Missing support for batch shape checking

Most activations will work batch-wise with x.shape == (batch_size, features) due to NumPy broadcasting. However:
    You don't verify or enforce that x is either 1D or 2D.
    This might break for odd shapes (e.g., 3D tensors for ConvNets in the future).
Suggestion:
Add shape checks or reshape handling logic if you plan to extend this to CNNs or batched models.
'''

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


# TODO Theres a possible overflow in __call__(), handle it!
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

'''
TODO softmax returns the Jacobian Matrix, but most other derivatives returns vectors, checkout for inconsistency , possible fixes:
    Document clearly: Softmax.derivative() returns a Jacobian.
    Consider a batched=True/False flag.
    Alternatively, return only the diagonal for simplicity (approximate) if consistency is more important than precision.

TODO Incorrect derivative of Softmax for training use:
    In cross-entropy + softmax combinations, you're typically supposed to avoid computing the Jacobian manually.
    If you're using cross-entropy loss with Softmax, the gradient simplifies greatly:
    softmax_output - one_hot_labels
    So you don't need the Jacobian unless you're computing pure Softmax derivatives without a loss function.
    Suggestion: Include a note in the docstring or code to clarify that Softmax().derivative() is not typically used directly during backprop.
'''
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