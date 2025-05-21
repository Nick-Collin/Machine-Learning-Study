from ninc.Neural_Network import *
from ninc.DataHandling import Dataset

class Optimizer:
    def update(self, layer: Layer, learning_rate: float) -> None:
        raise NotImplementedError("This method should be implemented by subclasses!")
    
class SGD(Optimizer):
    def update(self, 
               layer:Layer, 
               learning_rate: float) -> None:
        
        assert hasattr(layer, "gradient"), "Layer must have a gradient attribute. Maybe it was not calculated?"
        assert hasattr(layer, "weights"), "Layer must have a weights attribute. Maybe it was not initialized?"
        layer.weights -= learning_rate * layer.gradient
        layer.bias -= learning_rate * layer.deltas
class PolyakMomentum(Optimizer):
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum

    def update(self, 
               layer: Layer, 
               learning_rate: float) -> None:
        
        if not hasattr(layer, 'velocity_weights'):
            layer.velocity_weights = np.zeros_like(layer.weights)
        if not hasattr(layer, 'velocity_bias'):
            layer.velocity_bias = np.zeros_like(layer.bias)
        
        # Update velocities
        layer.velocity_weights = self.momentum * layer.velocity_weights - learning_rate * layer.gradient
        layer.velocity_bias = self.momentum * layer.velocity_bias - learning_rate * layer.deltas
        
        # Update weights and biases
        layer.weights += layer.velocity_weights
        layer.bias += layer.velocity_bias

class RMSProp(Optimizer):
    def __init__(self, 
                 beta: float = 0.9, 
                 epsilon: float = 1e-8):
        
        self.beta = beta
        self.epsilon = epsilon

    def update(self, layer: Layer, learning_rate: float) -> None:
        if not hasattr(layer, 'v_weights'):
            layer.v_weights = np.zeros_like(layer.weights)
        if not hasattr(layer, 'v_bias'):
            layer.v_bias = np.zeros_like(layer.bias)
        
        # Update squared gradient estimates
        layer.v_weights = self.beta * layer.v_weights + (1 - self.beta) * (layer.gradient ** 2)
        layer.v_bias = self.beta * layer.v_bias + (1 - self.beta) * (layer.deltas ** 2)
        
        # Update weights and biases
        layer.weights -= learning_rate * layer.gradient / (np.sqrt(layer.v_weights) + self.epsilon)
        layer.bias -= learning_rate * layer.deltas / (np.sqrt(layer.v_bias) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep

    def update(self, layer: Layer, learning_rate: float) -> None:
        if not hasattr(layer, 'm_weights'):
            layer.m_weights = np.zeros_like(layer.weights)
            layer.v_weights = np.zeros_like(layer.weights)
            layer.m_bias = np.zeros_like(layer.bias)
            layer.v_bias = np.zeros_like(layer.bias)
            layer.t_adam = 0

        layer.t_adam += 1
        self.t += 1

        # Update moments for weights
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.gradient
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.gradient ** 2)

        # Update moments for bias
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.deltas
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.deltas ** 2)

        # Compute bias-corrected moments
        m_hat_weights = layer.m_weights / (1 - self.beta1 ** layer.t_adam)
        v_hat_weights = layer.v_weights / (1 - self.beta2 ** layer.t_adam)
        m_hat_bias = layer.m_bias / (1 - self.beta1 ** layer.t_adam)
        v_hat_bias = layer.v_bias / (1 - self.beta2 ** layer.t_adam)

        # Update weights and biases
        layer.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        layer.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
    
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
              shuffle: bool = True) -> None:

        if batched and not self.dataset.batches:
            self.dataset.split_batches(batch_size, shuffle)

        for epoch in range(self.epochs):
            if batched:
                for batch in self.dataset.batches:
                    self.train_step(batch.features, batch.labels)
            else:
                self.train_step(self.dataset.training.features, self.dataset.training.labels)
            
            if epoch % self.checkpoint == 0:
                logging.info(f"Epoch {epoch}/{self.epochs} completed. Processed {len(self.dataset.batches) if batched else len(self.dataset.training.features)} samples.")
                logging.debug(f"Current weights shapes: {[layer.weights.shape for layer in self.nn.layers]}")
                self.validate()

    def validate(self) -> None:
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
                    labels: Vector) -> None:
        
        for feature, label, in zip(features, labels):
            self.nn.propagate_forward(feature)
            self.nn.propagate_backward(label)
            for layer in self.nn.layers:
                self.optimizer.update(layer, self.learning_rate)
                logging.debug(f"Updated layer {layer.idx} with weights: {layer.weights}")
    
    def test(self):
        if not hasattr(self.dataset, "testing"):
            raise RuntimeError("Test dataset is not available.")

        features = self.dataset.testing.features
        labels = self.dataset.testing.labels

        total_loss = 0
        for feature, label in zip(features, labels):
            predictions = self.nn.propagate_forward(feature)
            total_loss += cost(predictions, label, derivative=False, mode=self.nn.loss_func)

        mean_loss = total_loss / len(features)
        logging.info(f"Test loss: {mean_loss.mean()}")
    
    def calculate_precision(self, dataset_type="testing"):
        if not hasattr(self.dataset, dataset_type):
            raise RuntimeError(f"{dataset_type.capitalize()} dataset is not available.")

        dataset = getattr(self.dataset, dataset_type)
        features = dataset.features
        labels = dataset.labels

        correct_predictions = 0
        total_predictions = len(features)

        for feature, label in zip(features, labels):
            predictions = self.nn.propagate_forward(feature)
            predicted_label = np.argmax(predictions)
            true_label = np.argmax(label)

            if predicted_label == true_label:
                correct_predictions += 1

        precision = (correct_predictions / total_predictions) * 100
        logging.info(f"Precision on {dataset_type} dataset: {precision:.2f}%")
        return precision
