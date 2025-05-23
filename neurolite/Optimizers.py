from neurolite.Neural_Network import Layer
import logging
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, 
               layer: Layer, 
               learning_rate: float) -> None:
        
        raise NotImplementedError("This method should be implemented by subclasses!")

    def _validate_layer(self, 
                        layer: Layer) -> None:
        
        if np.any(np.isnan(layer.weights)):
            raise ValueError("Weights exploded to NaN. Reduce learning rate.")

        if layer.weights is None or layer.bias is None:
            raise ValueError("Weights and bias must be initialized before optimization.")

        if not hasattr(layer, "gradient") or layer.gradient is None:
            logging.error("Layer must have a gradient attribute. Maybe it was not calculated?")
            raise AttributeError("Layer must have a gradient attribute. Maybe it was not calculated?")
        
        if not isinstance(layer.gradient, np.ndarray):
            raise TypeError("Expected gradient to be a NumPy array.")
        
        if not hasattr(layer, "deltas") or layer.deltas is None:
            logging.error("Layer must have deltas attribute.")
            raise AttributeError("Layer must have deltas attribute.")


    def _update_bias(self, 
                     layer: Layer, 
                     update: np.ndarray) -> None:
        
        if layer.deltas is not None:
            if update.shape == layer.bias.shape:
                layer.bias += update
            else:
                logging.warning(f"Layer {layer.idx} bias shape {layer.bias.shape} does not align with update shape {update.shape}")
        else:
            logging.warning(f"Layer {layer.idx} has no deltas — skipping bias update.")



class SGD(Optimizer):
    def update(self,
               layer: Layer,
               learning_rate: float) -> None:
               
        self._validate_layer(layer)
        
        layer.weights -= learning_rate * layer.gradient
        self._update_bias(layer, -1 * learning_rate * layer.deltas)
        


class PolyakMomentum(Optimizer):
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum

    def update(self,
               layer: Layer,
               learning_rate: float) -> None:
        
        self._validate_layer(layer)
        
        if not hasattr(layer, 'velocity_weights'):
            layer.velocity_weights = np.zeros_like(layer.weights)
        if not hasattr(layer, 'velocity_bias'):
            layer.velocity_bias = np.zeros_like(layer.bias)


        # Update velocities
        layer.velocity_weights = self.momentum * layer.velocity_weights - learning_rate * layer.gradient
        layer.velocity_bias = self.momentum * layer.velocity_bias - learning_rate * layer.deltas

        # Update weights and biases
        layer.weights += layer.velocity_weights
        self._update_bias(layer, layer.velocity_bias)

        


class RMSProp(Optimizer):
    def __init__(self,
                 beta: float = 0.9,
                 epsilon: float = 1e-8) -> None:

        self.beta = beta
        self.epsilon = epsilon

    def update(self, 
               layer: Layer, 
               learning_rate: float) -> None:
    
        self._validate_layer(layer)
        
        if not hasattr(layer, 'v_weights'):
            layer.v_weights = np.zeros_like(layer.weights)
        if not hasattr(layer, 'v_bias'):
            layer.v_bias = np.zeros_like(layer.bias)


        # Update squared gradient estimates
        layer.v_weights = self.beta * layer.v_weights + (1 - self.beta) * (layer.gradient ** 2)
        layer.v_bias = self.beta * layer.v_bias + (1 - self.beta) * (layer.deltas ** 2)

        # Update weights and biases
        layer.weights -= learning_rate * layer.gradient / (np.sqrt(layer.v_weights) + self.epsilon)

        self._update_bias(layer, -1 * learning_rate * layer.deltas / (np.sqrt(layer.v_bias) + self.epsilon))

        


class Adam(Optimizer):
    def __init__(self, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8) -> None:
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep

    def update(self, layer: Layer, 
               learning_rate: float) -> None:
    
        self._validate_layer(layer)

        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_bias = np.zeros_like(layer.bias)
            layer.v_bias = np.zeros_like(layer.bias)
            logging.debug(f"Adding Adam m_weights to layer: {layer.idx}")

        
        self.t += 1

        # Update moments for weights
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.gradient
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.gradient ** 2)

        # Update moments for bias
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.deltas
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.deltas ** 2)

        # Compute bias-corrected moments
        m_hat_weights = layer.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = layer.v_weights / (1 - self.beta2 ** self.t)
        m_hat_bias = layer.m_bias / (1 - self.beta1 ** self.t)
        v_hat_bias = layer.v_bias / (1 - self.beta2 ** self.t)

        # Update weights 
        layer.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        self._update_bias(layer, -1 * learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon))
