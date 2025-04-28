import numpy as np

#Definig vector and matrix types
type Vector = np.ndarray
type Matrix = np.ndarray

#Defining a perceptron
class Perceptron:
    def __init__(self, layer: str = "not provided", num: str = "not provided"):
        #Debug only
        self.layer = layer
        self.num = num


    #TODO Implement more activation methods
    def activate(self, inputs:Vector, weights: Vector, bias: float, activation: str, verbose: bool = False) -> float:

        #Making sure everything was correctly passed
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {layer}, perceptron: {num}"
        assert isinstance(weights, np.ndarray), f"expected a weight vector at layer: {layer}, perceptron: {num}"
        assert inputs.ndim == 1, "Inputs must be  a vector"
        assert weights.ndim == 1, "Weights must be a vector"
        assert inputs.shape == weights.shape, "Inputs and weights mismatch shapes"

        weighted_sum = np.dot(inputs, weights) + bias
        if verbose: print(f"sum = inputs({inputs}) . weights({weights}) + bias({bias}) =  {weighted_sum}, dot = {np.dot(inputs, weights)}")

        match activation:
            case "sigmoid":
                return 1 / (1 + np.exp(-weighted_sum))
            case "relu":
                return max(0, weighted_sum)
            case "tanh":
                return np.tanh(weighted_sum)
            case _:
                return weighted_sum  # Linear
        

#Defining a layer
class Layer:
    def __init__(self, inputs: Vector, weights: Matrix, num_of_perceptrons: int, activation_method: str = "", num: str = "not provided"):
        
        #Making sure everything was correctly initialized
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {num}"
        assert isinstance(weights, np.ndarray), f"expected a weight matrix at layer: {num}"
        assert weights.ndim == 2, f"Expected 2D matrix weight, got {weights.ndim}D - Layer: {num}"
        assert weights.shape[0] == num_of_perceptrons, "Number of weight rows must match number of perceptrons"
        assert weights.shape[1] == len(inputs) + 1, f"Each weight vector must match input length + 1 for bias - Layer: {num}"
        assert inputs.ndim == 1, f"Expected 1D Vector input, got {inputs.ndim}D Matrix - Layer: {num}"

        #Debug only
        self.num = num
        
        self.inputs = inputs
        #Note that the W[k,j] matrix should have an extra j column with the biases for each unit k on the layer
        self.weights = weights


        self.perceptrons = []
        for k in range(num_of_perceptrons):
            #Here we limit the weight vector to not include the last scalar because it represents the bias
            h =  Perceptron(self.num, f"{k}")
            self.perceptrons.append(h)

        self.activation_method = activation_method
    
    def get_out(self, verbose: bool = False) -> Vector:
        out = []
        k=0
        for perceptron in self.perceptrons:
            #Important note: here is crucial to update the perceptron inputs, because elsewhere it would keep with the inputs set at init time before the foward propagation
            a = perceptron.activate(inputs= self.inputs, weights= self.weights[k][:-1], bias= self.weights[k][-1], activation= self.activation_method, verbose= verbose)
            out.append(a)
            k += 1
            
            #Debug
            if verbose : print(f"calculating layer {self.num}, unit {perceptron.num}, method: {self.activation_method}, a = {a}, current out: {out}, inputs: {self.inputs}, weights: {self.weights}")

        if verbose: print(f"out: {out}")
        return np.array(out)

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, input_data: Vector, verbose: bool = False) -> Vector:
        for layer in self.layers:
            layer.inputs = input_data
            if verbose: print(f"current layer: {layer.num}, inputs: {layer.inputs}")
            input_data = layer.get_out(verbose)
        return input_data


def lossvec(output: Vector, expected: Vector) -> Vector:
    return np.square(np.subtract(output, expected))

def loss(output:Vector, expected: Vector) -> float:
    return np.sum(lossvec(output, expected))


