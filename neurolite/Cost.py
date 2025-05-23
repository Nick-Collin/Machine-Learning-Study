from neurolite.Util import Vector
from abc import ABC, abstractmethod
import logging
import numpy as np

class CostFunction(ABC):
    @abstractmethod
    def __call__(self, 
                 x: Vector, 
                 y: Vector) -> Vector:
        pass

    @abstractmethod
    def derivative(self, 
                   x: Vector, 
                   y: Vector) -> Vector:
        pass


class MSE(CostFunction):
    def __call__(self, 
                 x: Vector, 
                 y: Vector) -> Vector:
        
        return (x - y) ** 2

    def derivative(self, 
                   x: Vector, 
                   y: Vector) -> Vector:
        
        return 2 * (x - y)


class CrossEntropy(CostFunction):
    def __call__(self, 
                 x: Vector, 
                 y: Vector) -> Vector:

        if not np.all((y == 0) | (y == 1)):
            raise ValueError("CrossEntropy expects binary (0 or 1) labels.")
        
        # Clip x to avoid log(0)
        eps = 1e-12
        x_clipped = np.clip(x, eps, 1 - eps)
        return -np.sum(y * np.log(x_clipped) + (1 - y) * np.log(1 - x_clipped), axis=-1)

    def derivative(self, 
                   x: Vector, 
                   y: Vector) -> Vector:
                   
        eps = 1e-12
        x_clipped = np.clip(x, eps, 1 - eps)

        #Make shure denominator is not zero
        denom = x_clipped * (1 - x_clipped) + eps
        denom[denom == 0] = eps

        return (x_clipped - y) / denom


# Registry for built-in and custom cost functions
_cost_registry = {
    'MSE': MSE(),
    'CrossEntropy': CrossEntropy(),
}

def register_cost(name: str, 
                  cost: CostFunction,
                  overwrite: bool = False):
    
    if name in _cost_registry and not overwrite:
        raise ValueError(f"Cost function '{name}' already exists. Use `overwrite=True` to replace it.")
    
    _cost_registry[name] = cost

def get_cost(mode: str | CostFunction) -> CostFunction:

    if isinstance(mode, CostFunction):
        return mode
    
    if mode in _cost_registry:
        return _cost_registry[mode]
    
    raise ValueError(f"No cost function named: {mode}. Use one of {list(_cost_registry.keys())} or register your own.")

def cost(x: Vector, 
         y: Vector, 
         derivative: bool= False, 
         mode: str | CostFunction= "MSE") -> Vector:
    
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{ x.shape } vs y{ y.shape }")

    try:
        cost_func = get_cost(mode)
        logging.debug(f"Cost function '{cost_func.__class__.__name__}' called with input shapes x: {x.shape}, y: {y.shape}")

        if derivative:
            return cost_func.derivative(x, y)
        
        else:
            return cost_func(x, y)
        
    except Exception as e:
        logging.error(f"Error in cost function '{mode}': {e}")
        raise