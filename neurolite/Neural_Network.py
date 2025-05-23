# Imports
from neurolite.Util import Vector, Matrix, is_vector, is_matrix
from neurolite.DataHandling import Dataset
from neurolite.Activation import activation
from neurolite.Cost import cost
import logging
import numpy as np

class NeuralNetwork:
    """
    Represents a feedforward neural network composed of multiple layers.
    Handles forward and backward propagation, saving/loading, and layer management.
    """
    def __init__(self, 
                 dataset: Dataset,
                 layers: list["Layer"], 
                 loss_func: str = "MSE"):
        """
        Initialize the neural network with a dataset, list of layers, and loss function.
        Ensures all layers are properly initialized and weight dimensions match.
        """

        assert hasattr(dataset, "training"), "Dataset must be splited and not empty before nn initialization"

        self.layers: list[Layer] = layers
        self.loss_func: str = loss_func
        self.dataset: Dataset = dataset
        self.input_size = self.dataset.training.features[0].shape[0]


        for idx, layer in enumerate(self.layers):
            layer.parent_nn = self
            layer.idx = idx
            # Initialize weights if not already set
            if layer.weights is None:
                layer.initialize_weights(mode=layer.initialization_mode)
        # Check weight dimensions for all layers
        for i in range(1, len(layers)):
            prev_units = layers[i-1].num_of_units
            assert layers[i].weights.shape[1] == prev_units, \
                f"Layer {i} weight dim mismatch. Expected {prev_units}, got {layers[i].weights.shape[1]}"
    
    def propagate_forward(self, 
                          inputs: Vector) -> Vector:
        """
        Perform a forward pass through all layers of the network.
        Returns the output vector.
        """
        logging.debug(f"Forwarding with inputs: {inputs}")
        try:
            x = inputs.copy()
            for layer in self.layers:
                x = layer.forward_pass(x)
            return x
        
        except Exception as e:
            logging.error(f"Error during forward propagation: {e}")
            raise
    
    '''
    TODO The propagate_backward() method assumes last layer always has activation and cost
        output_layer.deltas = cost(...) * activation(...)
        This assumes:
            Activation derivative exists for last layer.
            Cost + activation combo is valid (e.g., MSE + sigmoid may be inefficient).
        Improvement: Add logic to handle softmax + cross-entropy combined derivative as a special case, or allow user-defined derivative computation for output layers.
    '''
    def propagate_backward(self, 
                           expected: Vector):
        """
        Perform a backward pass (backpropagation) to compute gradients for all layers.
        """
        try:
            output_layer = self.layers[-1]
            output_layer.deltas = cost(x= output_layer.a, y= expected, derivative= True, mode= self.loss_func) * activation(x= output_layer.z, derivative= True, mode=output_layer.activation_mode)
            output_layer.get_gradient()
            for layer in reversed(self.layers[:-1]):
                layer.get_deltas(next= self.layers[layer.idx + 1])
                layer.get_gradient()
        except Exception as e:
            logging.error(f"Error during backward propagation: {e}")
            raise

    def save(self, 
             filepath: str):
        """
        Save model weights, biases, and architecture metadata to a .npz file.
        - Weights and biases for each layer are saved with unique keys.
        - Architecture (layer configuration, activation, etc.) and loss function are saved as JSON metadata.
        - The metadata allows full reconstruction of the model when loading.
        """
        import json
        params = {}
        # Save weights and biases
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_weights"] = layer.weights
            params[f"layer_{i}_bias"] = layer.bias
        # Save architecture and loss function as metadata
        architecture = []
        for layer in self.layers:
            architecture.append({
                'num_of_units': layer.num_of_units,
                'activation_mode': layer.activation_mode,
                'initialization_mode': layer.initialization_mode,
                'clipping': layer.clipping,
                'clip_size': layer.clip_size
            })
        metadata = {
            'architecture': architecture,
            'loss_func': self.loss_func
        }
        params['metadata'] = np.array([json.dumps(metadata)])
        try:
            np.savez(filepath, **params)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save model to {filepath}: {e}")
            raise

    @classmethod
    def load(cls, 
             filepath: str, 
             dataset: Dataset) -> "NeuralNetwork":
        """
        Load model weights, biases, and architecture metadata from a .npz file into a new NeuralNetwork instance.
        - Reads weights and biases for each layer from the file.
        - Loads architecture and loss function from JSON metadata.
        - Reconstructs each layer and the neural network as originally saved, so no manual architecture specification is needed.
        """
        import json
        try:
            data = np.load(filepath, allow_pickle=True)
            # Load metadata
            metadata = json.loads(data['metadata'][0])
            architecture = metadata['architecture']
            loss_func = metadata.get('loss_func', 'MSE')
            # Reconstruct layers
            layers = []
            for i, layer_info in enumerate(architecture):
                weights = data[f"layer_{i}_weights"]
                bias = data[f"layer_{i}_bias"]
                layer = Layer(
                    num_of_units=layer_info['num_of_units'],
                    activation_mode=layer_info['activation_mode'],
                    weights=weights,
                    bias=bias,
                    initialization_mode=layer_info.get('initialization_mode', 'Xavier'),
                    clipping=layer_info.get('clipping', True),
                    clip_size=layer_info.get('clip_size', 500)
                )
                layers.append(layer)
            nn = cls(dataset=dataset, layers=layers, loss_func=loss_func)
            logging.info(f"Model loaded from {filepath}")
            return nn
        except Exception as e:
            logging.error(f"Failed to load model from {filepath}: {e}")
            raise
        

class Layer:
    """
    Represents a single layer in a neural network, including weights, biases, activation, and gradient logic.
    """
    def __init__(self, 
                 num_of_units: int,
                 activation_mode: str= "sigmoid",
                 weights: Matrix= None,
                 bias: Vector= None,
                 initialization_mode: str= "Xavier",
                 clipping: bool = True,
                 clip_size: int = 500):
        """
        Initialize a layer with the given number of units, activation, and initialization settings.
        """
        assert isinstance(initialization_mode, str), f"initialization_mode must be a string, got {type(initialization_mode)} instead."
        assert isinstance(num_of_units, int), f"num_of_units must be an integer, got {type(num_of_units)} instead."
        assert num_of_units > 0, "num_of_units must be greater than 0"
        assert isinstance(activation_mode, str), f"activation_mode must be a string, got {type(activation_mode)} instead."
        assert bias is None or is_vector(bias), f"bias must be a Vector, got {type(bias)} instead."
        if weights is not None:
            assert is_matrix(weights), f"Weights must be a Matrix, got {type(weights)} instead."

        self.parent_nn: NeuralNetwork
        self.idx: int
        self.z: Vector
        self.a: Vector
        self.deltas: Vector
        self.gradient: Matrix 
        self.inputs: Vector
        self.num_of_units = num_of_units
        self.initialization_mode = initialization_mode
        self.weights: Matrix = None
        if weights is not None:
            self.weights: Matrix = weights 

        self.bias = bias if bias is not None else np.zeros(num_of_units)
        self.activation_mode: str = activation_mode
        self.clipping = clipping
        self.clip_size = clip_size

    #TODO Create a module for a class based implementation of weight initialization
    def initialize_weights(self, mode: str = "Xavier"):
        """
        Initialize the weights and biases for this layer using the specified mode.
        Supported modes: Xavier, XavierNormal, He, HeNormal.
        """
        assert len(self.parent_nn.dataset.training.features) > 0, "Training data is empty"
        
        if self.idx == 0:
            sample_input = self.parent_nn.dataset.training.features[0]
            n_in = sample_input.shape[0]

        else:
            previous_layer = self.parent_nn.layers[self.idx - 1]
            n_in = previous_layer.num_of_units
        
        n_out = self.num_of_units

        if mode == "Xavier":
            # Xavier uniform initialization
            limit = np.sqrt(6 / (n_in + n_out))
            self.weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

        elif mode == "XavierNormal":
            # Xavier normal initialization
            std = np.sqrt(2 / (n_in + n_out))
            self.weights = np.random.normal(0, std, size=(n_out, n_in))

        elif mode == "He":
            # He uniform initialization
            limit = np.sqrt(6 / n_in)
            self.weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

        elif mode == "HeNormal":
            # He normal initialization
            std = np.sqrt(2 / n_in)
            self.weights = np.random.normal(0, std, size=(n_out, n_in))

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        self.bias = np.zeros(n_out)

    def forward_pass(self, inputs: Vector) -> Vector:
        """
        Compute the output of this layer given input vector, applying weights, bias, and activation.
        Optionally clips the pre-activation values to avoid overflow.
        """
        assert isinstance(inputs, Vector), f"Inputs must be a numpy Vector, got {type(inputs)} instead."
        assert inputs.ndim == 1, f"Inputs must be a Vector ndarray, got {inputs.ndim}D instead."
        assert inputs.shape[0] > 0, "Input size must be greater than 0"
        assert inputs.shape[0] == self.weights.shape[1], "Input size must match the number of weights in the layer"
        assert self.weights is not None, "Weights must be initialized before forward pass"
        assert self.bias is not None, "Bias must be initialized before forward pass"

        # Compute pre-activation (z) and apply activation function
        self.z = np.dot(self.weights, inputs) + self.bias
        if self.clipping:
            self.z = np.clip(self.z, -self.clip_size, self.clip_size)
        self.a = activation(self.z, mode= self.activation_mode)
        self.inputs = inputs

        logging.debug(f"Forwarding with inputs of shape: {inputs.shape}")

        return self.a

    def get_deltas(self, next: "Layer") -> Vector:
        """
        Compute the delta (error term) for this layer based on the next layer's weights and deltas.
        Used for backpropagation.
        """
        assert isinstance(next, Layer), "Next layer must be an instance of Layer"
        assert next.weights is not None, "Next layer must have weights initialized"
        assert next.deltas is not None, "Next layer must have deltas initialized"
        assert next.deltas.shape[0] == next.weights.shape[0], "Next layer deltas must match the number of neurons in the next layer"
        assert next.weights.shape[1] == self.num_of_units, "Next layer weights must match the number of neurons in this layer"
        assert self.inputs is not None, "Inputs must be initialized before calculating gradient"

        if not self.idx == len(self.parent_nn.layers) - 1:
            # Delta for hidden layers
            self.deltas = np.dot(next.weights.T, next.deltas) * activation(self.z, derivative= True, mode= self.activation_mode)
            
        return self.deltas

    def get_gradient(self) -> Matrix:
        """
        Compute the gradient of the loss with respect to the weights for this layer.
        """
        assert self.deltas is not None, "Deltas must be initialized before calculating gradient"
        assert self.inputs is not None, "Inputs must be initialized before calculating gradient"

        # Gradient is the outer product of deltas and inputs
        self.gradient = np.outer(self.deltas, self.inputs)
        if self.clipping:
            np.clip(self.gradient, -self.clip_size, self.clip_size, out=self.gradient)

        return self.gradient
    