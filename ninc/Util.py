import numpy as np
import logging
import abc

Tensor = np.ndarray
Matrix = np.ndarray
Vector = np.ndarray

def is_tensor(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray)

def is_matrix(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 2

def is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1

class ActivationFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Vector) -> Vector:
        pass

    @abc.abstractmethod
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
}

def register_activation(name: str, activation: ActivationFunction):
    _activation_registry[name] = activation

def get_activation(mode: str | ActivationFunction) -> ActivationFunction:
    if isinstance(mode, ActivationFunction):
        return mode
    if mode in _activation_registry:
        return _activation_registry[mode]
    raise ValueError(f"No activation function named: {mode}. Use one of {list(_activation_registry.keys())} or register your own.")

def activation(x: Vector, derivative: bool = False, mode: str | ActivationFunction = "sigmoid") -> Vector:
    act = get_activation(mode)
    logging.debug(f"Activation function '{act.__class__.__name__}' called with input shape {x.shape}")
    if derivative:
        return act.derivative(x)
    else:
        return act(x)

class CostFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Vector, y: Vector) -> Vector:
        pass

    @abc.abstractmethod
    def derivative(self, x: Vector, y: Vector) -> Vector:
        pass

class MSE(CostFunction):
    def __call__(self, x: Vector, y: Vector) -> Vector:
        return (x - y) ** 2
    def derivative(self, x: Vector, y: Vector) -> Vector:
        return 2 * (x - y)

# Registry for built-in and custom cost functions
_cost_registry = {
    'MSE': MSE(),
}

def register_cost(name: str, cost: CostFunction):
    _cost_registry[name] = cost

def get_cost(mode: str | CostFunction) -> CostFunction:
    if isinstance(mode, CostFunction):
        return mode
    if mode in _cost_registry:
        return _cost_registry[mode]
    raise ValueError(f"No cost function named: {mode}. Use one of {list(_cost_registry.keys())} or register your own.")

def cost(x: Vector, y: Vector, derivative: bool= False, mode: str | CostFunction= "MSE") -> Vector:
    cost_func = get_cost(mode)
    logging.debug(f"Cost function '{cost_func.__class__.__name__}' called with input shapes x: {x.shape}, y: {y.shape}")
    if derivative:
        return cost_func.derivative(x, y)
    else:
        return cost_func(x, y)

def norm_input(x: Vector, feature_mean: Vector, feature_std: Vector, norm_mode: str) -> Vector:
    match norm_mode:
        case "standardization":
            return (x - feature_mean) / feature_std
        
        case _:
            raise AttributeError(f"Couldnt resolve norm mode")

def denorm_output(y: Vector, label_mean: float, label_std: float, norm_mode: str) -> Vector:
    assert label_mean is not None and label_std is not None, "Labels where not normalized! You shoudnt denormalize outputs"
    
    match norm_mode:
        case "standardization":
            return y * label_std + label_mean

