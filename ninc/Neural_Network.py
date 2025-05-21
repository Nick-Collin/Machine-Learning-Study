# Imports
from ninc.Util import *
from ninc.DataHandling import Dataset


class NeuralNetwork:
    def __init__(self, 
                 dataset: Dataset,
                 layers: list["Layer"], 
                 loss_func: str = "MSE"):
        
        
        self.layers: list[Layer] = layers
        self.loss_func: str = loss_func
        self.dataset: Dataset = dataset


        for idx, layer in enumerate(self.layers):
            layer.parent_nn = self
            layer.idx = idx
            
            if layer.weights == None:
                layer.initialize_weights(mode= layer.initialization_mode)
                
        for i in range(1, len(layers)):
            prev_units = layers[i-1].num_of_units
            layers[i].weights.shape[1] == prev_units
       

    def propagate_forward(self, inputs: Vector) -> Vector:
        logging.debug(f"Forwarding with inputs: {inputs}")
        
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)

        return inputs
    
    
    def propagate_backward(self, expected: Vector):
        output_layer = self.layers[-1]
        output_layer.deltas = cost(x= output_layer.a, y= expected, derivative= True, mode= self.loss_func) * activation(x= output_layer.z, derivative= True, mode=output_layer.activation_mode)
        output_layer.get_gradient()
        for layer in reversed(self.layers[:-1]):
            layer.get_deltas(next= self.layers[layer.idx + 1])
            layer.get_gradient()

    def save(self, filepath: str):
        """Save model weights and biases to a .npz file."""
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_weights"] = layer.weights
            params[f"layer_{i}_bias"] = layer.bias
        np.savez(filepath, **params)

    @classmethod
    def load(cls, filepath: str, dataset: Dataset, layers: list["Layer"], loss_func: str = "MSE") -> "NeuralNetwork":
        """Load model weights and biases from a .npz file into a new NeuralNetwork instance."""
        nn = cls(dataset=dataset, layers=layers, loss_func=loss_func)
        data = np.load(filepath)
        for i, layer in enumerate(nn.layers):
            layer.weights = data[f"layer_{i}_weights"]
            layer.bias = data[f"layer_{i}_bias"]
        return nn


class Layer:
    def __init__(self, 
                 num_of_units: int,
                 activation_mode: str= "sigmoid",
                 weights: Matrix= None,
                 bias: Vector= None,
                 initialization_mode: str= "Xavier",
                 cliping: bool = True,
                 clip_size: int = 500):
                 
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
        self.inputs = Vector
        self.num_of_units = num_of_units
        self.initialization_mode = initialization_mode
        self.weights: Matrix = None
        if weights is not None:
            self.weights: Matrix = weights 

        self.bias = bias if bias is not None else np.zeros(num_of_units)
        self.activation_mode: str = activation_mode
        self.cliping = cliping
        self.clip_size = clip_size

    def initialize_weights(self, mode: str = "Xavier"):
        if self.idx == 0:
            sample_input = self.parent_nn.dataset.training.features[0]
            n_in = sample_input.shape[0]

        else:
            previous_layer = self.parent_nn.layers[self.idx - 1]
            n_in = previous_layer.num_of_units
        
        n_out = self.num_of_units

        if mode == "Xavier":
            limit = np.sqrt(6 / (n_in + n_out))
            self.weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

        elif mode == "XavierNormal":
            std = np.sqrt(2 / (n_in + n_out))
            self.weights = np.random.normal(0, std, size=(n_out, n_in))

        elif mode == "He":
            limit = np.sqrt(6 / n_in)
            self.weights = np.random.uniform(-limit, limit, size=(n_out, n_in))

        elif mode == "HeNormal":
            std = np.sqrt(2 / n_in)
            self.weights = np.random.normal(0, std, size=(n_out, n_in))

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        self.bias = np.zeros(n_out)

    def forward_pass(self, inputs: Vector) -> Vector:
        assert isinstance(inputs, Vector), f"Inputs must be a numpy Vector, got {type(inputs)} instead."
        assert inputs.ndim == 1, f"Inputs must be a Vector ndarray, got {inputs.ndim}D instead."
        assert inputs.shape[0] > 0, "Input size must be greater than 0"
        assert inputs.shape[0] == self.weights.shape[1], "Input size must match the number of weights in the layer"
        assert self.weights is not None, "Weights must be initialized before forward pass"
        assert self.bias is not None, "Bias must be initialized before forward pass"

        #TODO think about this operation, if its output are ordered
        self.z = np.dot(self.weights, inputs) + self.bias
        if self.cliping:
            self.z = np.clip(self.z, -self.clip_size, self.clip_size)
        self.a = activation(self.z, mode= self.activation_mode)
        self.inputs = inputs

        logging.debug(f"Forwarding with inputs of shape: {inputs.shape}")

        return self.a

    def get_deltas(self, next: "Layer") -> Vector:
        assert isinstance(next, Layer), "Next layer must be an instance of Layer"
        assert next.weights is not None, "Next layer must have weights initialized"
        assert next.deltas is not None, "Next layer must have deltas initialized"
        assert next.deltas.shape[0] == next.weights.shape[0], "Next layer deltas must match the number of neurons in the next layer"
        assert next.weights.shape[1] == self.num_of_units, "Next layer weights must match the number of neurons in this layer"
        assert self.inputs is not None, "Inputs must be initialized before calculating gradient"

        if not self.idx == len(self.parent_nn.layers) - 1:
            self.deltas = np.dot(next.weights.T, next.deltas) * activation(self.z, derivative= True, mode= self.activation_mode)
            
        return self.deltas

    def get_gradient(self) -> Matrix:
        assert self.deltas is not None, "Deltas must be initialized before calculating gradient"
        assert self.inputs is not None, "Inputs must be initialized before calculating gradient"

        #shape npers x ninputs
        self.gradient = np.outer(self.deltas, self.inputs)
        return self.gradient