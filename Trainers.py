"""
This module contains classes and methods for training neural networks.

Classes:
    Trainer: Handles the training process of a neural network.
    Optimizer: Base class for optimization algorithms.
    SGD: Implements the Stochastic Gradient Descent optimization algorithm.
    PolyakMomentum: Implements the Polyak Momentum optimization algorithm.

Methods:
    Trainer.train(dataset: Dataset): Trains the neural network using the provided dataset.
    Optimizer.update(layer: Layer, learning_rate: float): Updates the weights of a layer based on the optimization algorithm.
"""

from Neural_Network import *
import logging

# Setup logging
logger = logging.getLogger("Trainer")

class Optimizer:
    """
    Base class for optimization algorithms.

    Methods:
        update(layer: Layer, learning_rate: float): Updates the weights of a layer based on the optimization algorithm.
    """
    def update(self, layer: Layer, learning_rate: float):
        #TODO Validate that the layer and learning_rate are valid inputs
        #TODO Add error handling for invalid layer attributes
        raise NotImplementedError("This method should be implemented by subclasses!")
    
class SGD(Optimizer):
    """
    Implements the Stochastic Gradient Descent optimization algorithm.

    Methods:
        update(layer: Layer, learning_rate: float): Updates the weights of a layer using SGD.
    """
    def update(self, layer:Layer, learning_rate: float):
        layer.weights -= learning_rate * layer.gradient

class PolyakMomentum(Optimizer):
    """
    Implements the Polyak Momentum optimization algorithm.

    Methods:
        update(layer: Layer, learning_rate: float): Updates the weights of a layer using Polyak Momentum.
    """
    def __init__(self):
        pass #TODO

    def update(self, layer: Layer, learning_rate: float):
        #TODO Validate that the layer and learning_rate are valid inputs
        #TODO Add error handling for invalid layer attributes
        pass #TODO
    
    
class Trainer:
    """
    Handles the training process of a neural network.

    Attributes:
        nn (NeuralNetwork): The neural network to be trained.
        optimizer (Optimizer): The optimization algorithm to use.
        learning_rate (float): The learning rate for training.
        epochs (int): The number of training epochs.
        batches (Data): The data batches for training.

    Methods:
        train(dataset: Dataset): Trains the neural network using the provided dataset.
    """
    def __init__(self,
                 nn: NeuralNetwork,
                 optimizer: Optimizer,
                 epochs: int,
                 learning_rate: float): 
        
        #TODO Assert args are valid

        self.nn: NeuralNetwork = nn
        self.optimizer: Optimizer = optimizer
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.dataset: Dataset = nn.dataset
        
    
    def train(self,
              batched: bool = True,
              batch_size: int = 32,
              shuffle: bool = True):

        if batched and not self.dataset.batches:
            self.dataset.split_batches(batch_size, shuffle)

        for epoch in range(self.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            if batched:
                for batch in self.dataset.batches:
                    self.train_step(batch.features, batch.labels)

            else:
                self.train_step(self.dataset.training.features, self.dataset.training.labels)

    def train_step(self, 
                    features: Vector,    
                    labels: Vector):
        
        # Validate features and labels
        if not is_tensor(features) or not is_vector(labels):
            logger.error("Invalid features or labels for training step.")
            raise ValueError("Features must be a tensor and labels must be a vector.")

        for feature, label, in zip(features, labels):
            self.nn.propagate_forward(feature)
            self.nn.propagate_backward(label)
            for layer in self.nn.layers:
                self.optimizer.update(layer, self.learning_rate)
                logger.debug(f"Updated weights for layer {layer.idx}")