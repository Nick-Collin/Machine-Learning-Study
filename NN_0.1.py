import numpy as np

#Definig vector and matrix types
type Vector = np.ndarray
type Matrix = np.ndarray

#Defining a perceptron
class Perceptron:
    def __init__(self, inputs: Vector, weights: Vector, bias: float, layer: str = "not provided", num: str = "not provided"):
        
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {layer}, perceptron: {num}"
        assert isinstance(weights, np.ndarray), f"expected a weight vector at layer: {layer}, perceptron: {num}"
        assert inputs.ndim == 1, "Inputs must be  a vector"
        assert weights.ndim == 1, "Weights must be a vector"
        assert inputs.shape == weights.shape, "Inputs and weights mismatch shapes"

        #Debug only
        self.layer = layer
        self.num = num

        self.bias = bias
        self.weights = weights
        self.inputs = inputs


    #TODO Implement more activation methods
    def activate(self, activation: str, verbose: bool = False) -> float:
        weighted_sum = np.dot(self.inputs, self.weights) + self.bias
        if verbose: print(f"sum = inputs({self.inputs}) . weights({self.weights}) + bias({self.bias}) =  {weighted_sum}, dot = {np.dot(self.inputs, self.weights)}")
        match activation:
            case "sigmoid":
                return 1 / (1 + np.exp(-weighted_sum))
            case _:
                return self.weighted_sum
            
        

#Defining a layer
class Layer:
    def __init__(self, inputs: Vector, weights: Matrix, num_of_perceptrons: int, activation_method: str = "", num: str = "not provided"):
        
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {num}"
        assert isinstance(weights, np.ndarray), f"expected a weight matrix at layer: {num}"
        assert weights.ndim == 2, f"Expected 2D matrix weight, got {x.ndim}D - Layer: {num}"
        assert weights.shape[0] == num_of_perceptrons, "Number of weight rows must match number of perceptrons"
        assert weights.shape[1] == len(inputs) + 1, f"Each weight vector must match input length + 1 for bias - Layer: {num}"
        assert inputs.ndim == 1, f"Expected 1D Vector input, got {x.ndim}D Matrix - Layer: {num}"

        #Debug only
        self.num = num
        
        self.inputs = inputs
        #Note that the W[k,j] matrix should have an extra j column with the biases for each unit k on the layer
        self.weights = weights


        self.perceptrons = []
        for k in range(num_of_perceptrons):
            #Here we limit the weight vector to not include the last scalar because it represents the bias
            h =  Perceptron(self.inputs, self.weights[k][:-1], self.weights[k][-1], self.num, f"{k}")
            self.perceptrons.append(h)

        self.activation_method = activation_method
    
    def get_out(self, verbose: bool = False) -> Vector:
        out = []
        for perceptron in self.perceptrons:
            a = perceptron.activate(self.activation_method, verbose)
            out.append(a)
            
            #Debug
            if verbose : print(f"calculating layer {self.num}, unit {perceptron.num}, method: {self.activation_method}, a = {a}, current out: {out}, inputs: {self.inputs}, weights: {self.weights}")
        return np.array(out)



def lossvec(output: Vector, expected: Vector) -> Vector:
    return np.square(np.subtract(output, expected))

def loss(output:Vector, expected: Vector) -> float:
    return np.sum(lossvec(output, expected))


# Input
inputs = np.array([0.6, 0.1])

# Hidden layer weights (2 perceptrons, each has 2 weights + 1 bias)
hidden_weights = np.array([
    [0.5, -0.6, 0.1],
    [-0.3, 0.8, 0.2]
])

# Output layer weights (1 perceptron, takes 2 hidden outputs + 1 bias)
output_weights = np.array([
    [1.2, -1.1, 0.3]
])

# Expected output
expected = [0.9]

#Debuging
vb = True

# Build network
hidden_layer = Layer(inputs=inputs, weights=hidden_weights, num_of_perceptrons=2, activation_method="sigmoid", num="1")
hidden_output = hidden_layer.get_out(verbose= vb)

output_layer = Layer(inputs=hidden_output, weights=output_weights, num_of_perceptrons=1, activation_method="sigmoid", num="2")
output = output_layer.get_out(verbose= vb)

# Compute and show loss
print("Input:              ", inputs)
print("Hidden layer output:", hidden_output)
print("Final output:       ", output)
print("Expected:           ", expected)
print("Loss:               ", loss(output, expected))

