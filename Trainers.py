from Neural_Network import *
from DataHandeling import Dataset

class Optimizer:
    def update(self, layer: Layer, learning_rate: float):
        raise NotImplementedError("This method should be implemented by subclasses!")
    
class SGD(Optimizer):
    def update(self, layer:Layer, learning_rate: float):
        assert hasattr(layer, "gradient"), "Layer must have a gradient attribute. Maybe it was not calculated?"
        assert hasattr(layer, "weights"), "Layer must have a weights attribute. Maybe it was not initialized?"
        layer.weights -= learning_rate * layer.gradient

class PolyakMomentum(Optimizer):
    def __init__(self):
        pass

    def update(self, layer: Layer, learning_rate: float):
        pass
    
class Trainer:
    def __init__(self,
                 nn: NeuralNetwork,
                 optimizer: Optimizer,
                 epochs: int,
                 learning_rate: float,
                 checkpoint: int = 10): 
        
        self.nn: NeuralNetwork = nn
        self.optimizer: Optimizer = optimizer
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.dataset: Dataset = nn.dataset
        self.checkpoint = checkpoint
        
    def train(self,
              batched: bool = True,
              batch_size: int = 32,
              shuffle: bool = True):

        if batched and not self.dataset.batches:
            self.dataset.split_batches(batch_size, shuffle)

        for epoch in range(self.epochs):
            if batched:
                for batch in self.dataset.batches:
                    self.train_step(batch.features, batch.labels)
            else:
                self.train_step(self.dataset.training.features, self.dataset.training.labels)
            
            if epoch % self.checkpoint == 0:
                logging.info(f"Epoch {epoch}/{self.epochs} completed.")
                logging.debug(f"Current weights: {[layer.weights for layer in self.nn.layers]}")
                self.validate()

    def validate(self):
        if not hasattr(self.dataset, "validation"):
            raise RuntimeError("Validation dataset is not available.")
        
        features = self.dataset.validation.features
        labels = self.dataset.validation.labels
        
        total_loss = 0
        for feature, label in zip(features, labels):
            predictions = self.nn.propagate_forward(feature)
            total_loss += cost(predictions, label, derivative=False, mode=self.nn.loss_func)
        
        mean_loss = total_loss / len(features)
        logging.info(f"Validation loss: {mean_loss.mean()}")


    def train_step(self, 
                    features: Vector,    
                    labels: Vector):
        
        for feature, label, in zip(features, labels):
            self.nn.propagate_forward(feature)
            self.nn.propagate_backward(label)
            for layer in self.nn.layers:
                self.optimizer.update(layer, self.learning_rate)
                logging.debug(f"Updated layer {layer.idx} with weights: {layer.weights}")