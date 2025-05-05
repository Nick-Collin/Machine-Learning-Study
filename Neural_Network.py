"""
This module contains the classes and functions that define a Neural Network.

Classes:
    NeuralNetwork: Represents a neural network composed of multiple layers, capable of forward and backward propagation.
    Layer: Represents a layer in a neural network, containing perceptrons, weights, and biases.

Methods:
    NeuralNetwork.propagate_forward(inputs: Vector) -> Vector: Performs forward propagation through all layers of the network.
    NeuralNetwork.propagate_backward(): Performs backward propagation through all layers of the network.
    Layer.forward_pass(inputs: Vector) -> Vector: Computes the output of the layer by calculating the output of each perceptron.
    Layer.get_gradient(next: Layer) -> Matrix: Computes the gradient of the layer for backpropagation.
"""

# Imports
from Util import *

class NeuralNetwork:
    """
    Represents a neural network, which is composed of multiple layers and is capable of performing forward and backward propagation.

    Attributes:
        layers (list[Layer]): A list of layers that make up the neural network.

    Methods:
        propagate_forward(inputs: Vector) -> Vector:
            Performs forward propagation through all layers of the network.
        propagate_backward():
            Performs backward propagation through all layers of the network.
    """


    def __init__(self, 
                 dataset: Dataset,
                 layers: list["Layer"], 
                 loss_func: str = "MSE"):
        """
        Initializes the NeuralNetwork with a list of layers and a specified loss function.

        Args:
            layers (list[Layer]): A list of layers that make up the neural network.
            loss_func (str): The loss function to use for training (default is "MSE").
        """

        self.layers: list[Layer] = layers
        self.loss_func: str = loss_func
        self.dataset: Dataset = dataset

        for idx, layer in enumerate(self.layers):
            layer.parent_nn = self
            layer.idx = idx
            
            if layer.weights == None:
                layer.initialize_weights(mode= layer.initialization_mode)

    def propagate_forward(self, inputs: Vector) -> Vector:
        """
        Perform forward propagation through all layers of the network.

        Args:
            inputs (Vector): The input vector to the network.

        Returns:
            Vector: The output vector after forward propagation.
        """

        #TODO Validate that the inputs are valid numpy arrays
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)

        return inputs
    
    def propagate_backward(self, expected: Vector):
        """
        Perform backward propagation through all layers of the network.
        """
        self.layers[-1].deltas = cost(x= self.layers[-1].a, y= expected, derivative= True, mode= self.loss_func)
        for layer in reversed(self.layers[:-1]):
            layer.get_gradient(next= self.layers[layer.idx + 1])


class Layer:
    """
    Represents a layer in a neural network, which contains multiple perceptrons and their associated weights.

    Attributes:
        weights (Matrix): The weights connecting neurons in this layer to the previous layer.
        bias (float): The bias term added to the weighted sum of inputs.
        parent_nn (NeuralNetwork): The neural network to which this layer belongs.
        idx (int): The identifier number of this layer in its parent neural network.
        gradient (Matrix): The gradient of the layer for backpropagation.

    Methods:
        forward_pass(inputs: Vector) -> Vector:
            Computes the output of the layer by applying weights, bias, and activation function.
        get_gradient(next: Layer) -> Matrix:
            Computes the gradient of the layer for backpropagation.
    """


    def __init__(self, 
                 num_of_units: int,
                 activation_mode: str= "sigmoid",
                 weights: Matrix= None,
                 bias: Vector= None,
                 initialization_mode: str= "Xavier"):
        """
        Initializes the Layer with weights, bias, and metadata.
        """

        # TODO Assert self.weights is a 2D Numpy array and it does not have any invalid value in it. self.z, self.deltas and self.a are vectors
        #TODO Add error handling for invalid layer attributes
        
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

    def initialize_weights(self, mode: str = "Xavier"):
        """
        Initialize weights using the specified mode.
        Supported modes: "Xavier", "XavierNormal", "He", "HeNormal"
        """

        if self.idx == 0:
            # Assuming input features dimension is inferred from the dataset
            # Get one sample's feature size from the parent dataset
            sample_input = self.parent_nn.dataset.training.features[0]
            previous_layer_units = sample_input.shape[0]

        n_out, n_in = self.weights.shape

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

        # Bias can be zero or small noise
        self.bias = np.zeros(n_out)



    def forward_pass(self, inputs: Vector) -> Vector:
        """
        Compute the output of the layer by applying weights, bias, and activation function.

        Args:
            inputs (Vector): The input vector to the layer.

        Returns:
            Vector: The output vector after applying weights, bias, and activation function.
        """
        self.z = np.dot(self.weights, inputs) + self.bias
        self.a = activation(self.z)
        self.inputs = inputs #save it for gradient calculation

        return self.a


    def get_gradient(self, next: "Layer") -> Matrix:
        """
        Compute the gradient of the layer for backpropagation.

        Args:
            next (Layer): The next layer in the neural network.

        Returns:
            Matrix: The gradient of the layer.
        """
        #TODO assert next is a layer with already computed deltas and valid weights

        #deltas[i] represent the partial derivative of the cost in respect to self.z[i]
        #next.weights.T[j,i] connects  this layer's neuron[j] with next layer's neuron[i]
        #delta[j] = sum(next.weitghts(j,i) * next.deltas[i] for i in range next.weights[j]) * activation_derivatice(self.z)
        self.deltas = np.dot(next.weights.T, next.deltas) * activation(self.z, derivative= True, mode= self.activation_mode)

        #TODO assert inputs and deltas are valid
        self.gradient = np.outer(self.inputs, self.deltas)
        return self.gradient

