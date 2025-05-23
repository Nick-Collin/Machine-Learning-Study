from neurolite.Neural_Network import *
from neurolite.DataHandling import Dataset
from neurolite.Optimizers import Optimizer
from neurolite.Cost import cost

class Trainer:
    def __init__(self,
                 nn: NeuralNetwork,
                 optimizer: Optimizer,
                 epochs: int,
                 learning_rate: float,
                 checkpoint: int = 10,
                 early_stopping: bool = False,
                 patience: int = 10,
                 min_delta: float = 0.0):

        self.nn: NeuralNetwork = nn
        self.optimizer: Optimizer = optimizer
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.dataset: Dataset = nn.dataset
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self._best_val_loss = float('inf')
        self._validations_no_improve = 0
        self._best_weights = None

    def train(self,
              batched: bool = True,
              batch_size: int = 32,
              shuffle: bool = True) -> None:
        
        if self.dataset is None:
            raise ValueError("NeuralNetwork object must have a dataset assigned.")

        if batched and not self.dataset.batches:
            raise RuntimeError("Batches not created. Check batch size or call split_batches() before training.")


        for epoch in range(self.epochs):
            if batched:
                for batch in self.dataset.batches:
                    self.train_step(batch.features, batch.labels)

            else:
                self.train_step(self.dataset.training.features, self.dataset.training.labels)

            # Early stopping logic
            val_loss = None
            if epoch % self.checkpoint == 0:
                logging.info(f"Epoch {epoch}/{self.epochs} completed. Processed {len(self.dataset.batches) if batched else len(self.dataset.training.features)} samples.")
                logging.debug(f"Current weights shapes: {[layer.weights.shape for layer in self.nn.layers]}")
                val_loss = self.validate(return_loss=True)

            if self.early_stopping and val_loss is not None:
                if val_loss < self._best_val_loss - self.min_delta:
                    self._best_val_loss = val_loss
                    self._validations_no_improve = 0

                    # Save best weights
                    self._best_weights = [(layer.weights.copy(), layer.bias.copy()) for layer in self.nn.layers]

                else:
                    self._validations_no_improve += 1
                    logging.info(f"No improvement in validation loss for {self._validations_no_improve * self.checkpoint} epochs.")

                if self._validations_no_improve * self.checkpoint >= self.patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}. Best validation loss: {self._best_val_loss}")

                    if self._best_weights is not None:
                        for layer, (w, b) in zip(self.nn.layers, self._best_weights):
                            layer.weights = w
                            layer.bias = b
                    break

    def validate(self, 
                 return_loss=False) -> None:
        
        if not hasattr(self.dataset, "validation") or self.dataset.validation is None:
            raise RuntimeError("Validation dataset is not available.")
        

        
        features = self.dataset.validation.features
        labels = self.dataset.validation.labels
        total_loss = 0\
        
        if len(features) == 0:
            raise ValueError("Validation dataset is empty.")

        for feature, label in zip(features, labels):
            predictions = self.nn.propagate_forward(feature)
            total_loss += cost(predictions, label, derivative=False, mode=self.nn.loss_func)

        mean_loss = total_loss / len(features)
        logging.info(f"Validation loss: {mean_loss.mean()}")

        if return_loss:
            return mean_loss.mean()

    def train_step(self,
                    features: Vector,
                    labels: Vector) -> None:
        
        logging.debug(f"Train step: features shape ({features.shape}), labels shape ({labels.shape})")
        
        for feature, label, in zip(features, labels):
            if feature.shape[0] != self.nn.input_size:
                raise ValueError(f"Feature size mismatch: expected {self.nn.input_size}, got {feature.shape[0]}")
            
            try:
                out = self.nn.propagate_forward(feature)
                logging.debug(f"Train step: Out shape ({out.shape})")
                self.nn.propagate_backward(label)

                for layer in self.nn.layers:
                    if not hasattr(layer, "weights") or not hasattr(layer, "bias"):
                        raise AttributeError("Layer missing weights or bias attributes.")

                    if np.isnan(layer.weights).any():
                        raise FloatingPointError("NaNs detected in weights. Try lowering learning rate or using better initialization.")


                    self.optimizer.update(layer, self.learning_rate)
                    logging.debug(f"Updated layer {layer.idx} with weights: {layer.weights}")

            except Exception as e:
                logging.error(f"Error during train_step: {e}")
                raise