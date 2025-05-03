from Util import *

# Defining a perceptron
class Perceptron:
    def __init__(self, layer: str = "not provided", num: str = "not provided", activation_method: str = "Linear", verbose: bool = False):
        # Debug only
        self.layer = layer
        self.num = int(num)
        self.activation_method = activation_method
        self.verbose = verbose

    def activate(self, inputs: Vector, weights: Vector, bias: float, clip_size: int = 500) -> float:
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {self.layer}, perceptron: {self.num}"
        assert isinstance(weights, np.ndarray), f"expected a weight vector at layer: {self.layer}, perceptron: {self.num}"
        assert inputs.ndim == 1, "Inputs must be a vector"
        assert weights.ndim == 1, "Weights must be a vector"
        assert inputs.shape == weights.shape, "Inputs and weights mismatch shapes"

        self.weights, self.bias = weights, bias
        self.z = np.dot(inputs, weights) + bias
        if self.verbose:
            print(f"Perceptron {self.num} in Layer {self.layer}: inputs({inputs}), weights({weights}), bias({bias})")
            print(f"Sum (z) = {self.z}, dot product = {np.dot(inputs, weights)}")

        self.a = activation(self.z, mode=self.activation_method, clip_size= clip_size)
        if self.verbose:
            print(f"Activated output (a) = {self.a}")
        return self.a

    def get_deltas(self, next: "Layer" = None, expected: float = None, clip_size: float = 500) -> float:
        if next is None:
            # If there's no "next", we are at the output layer
            # Delta = cost derivative * activation derivative
            delta = cost(self.a, expected, derivative=True) * activation(self.z, mode=self.activation_method, derivative=True, clip_size= clip_size)
            if self.verbose:
                print(f"Perceptron {self.num} in Layer {self.layer} (Output Layer) - Delta: {delta}")
        else:
            assert next.deltas.ndim == next.weights.T[self.num].ndim, f"The next Layer's deltas vector must be the same dimension as the weight's vec associated with it"
            # If we have a next layer, we're in a hidden layer
            # Delta = (sum of next deltas * weights) * activation derivative
            delta = np.dot(next.deltas, next.weights.T[self.num]) * activation(self.z, mode=self.activation_method, derivative=True)
            if self.verbose:
                print(f"Perceptron {self.num} in Layer {self.layer} (Hidden Layer) - Delta: {delta}")
        
        # Save delta for future use
        self.delta = delta
        return delta


# Defining a layer
class Layer:
    def __init__(self, weights: Matrix, num_of_perceptrons: int, activation_method: str = "", num: str = "not provided", inputs: Vector = None, verbose: bool = False):
        if inputs is None:
            inputs = np.zeros(weights.shape[1] - 1)  # Initialize it as zeros
        
        # Making sure everything was correctly initialized
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {num}"
        assert isinstance(weights, np.ndarray), f"expected a weight matrix at layer: {num}"
        assert weights.ndim == 2, f"Expected 2D matrix weight, got {weights.ndim}D - Layer: {num}"
        assert weights.shape[0] == num_of_perceptrons, "Number of weight rows must match number of perceptrons"
        assert weights.shape[1] == len(inputs) + 1, f"Each weight vector must match input length + 1 for bias - Layer: {num}"
        assert inputs.ndim == 1, f"Expected 1D Vector input, got {inputs.ndim}D Matrix - Layer: {num}"

        # Debug only
        self.num = int(num)
        self.inputs = inputs
        self.weights = weights
        self.activation_method = activation_method
        self.perceptrons = []
        self.verbose = verbose

        for k in range(num_of_perceptrons):
            if verbose: print(f"Creating perceptron {k} in layer {self.num}")
            h = Perceptron(self.num, f"{k}", activation_method=self.activation_method, verbose= self.verbose)
            self.perceptrons.append(h)

    def get_out(self, verbose: bool = False, clip_size: float = 500) -> Vector:
        out = []
        k = 0
        for perceptron in self.perceptrons:
            # We limit the weight vector to not include the last scalar because it represents the bias
            # Important note: here is crucial to update the perceptron inputs, because elsewhere it would keep with the inputs set at init time before the forward propagation
            a = perceptron.activate(inputs=self.inputs, weights=self.weights[k][:-1], bias=self.weights[k][-1], clip_size= clip_size)
            out.append(a)
            k += 1

            # Debug
            if verbose:
                print(f"Layer {self.num} - Perceptron {perceptron.num}: activation = {a}, current output = {out}")

        if self.verbose:
            print(f"Layer {self.num} final output: {out}")
        return np.array(out)

    def get_deltas(self, next: "Layer" = None, expected: Vector = None, clip_size: float = 500) -> Vector:
        deltas = []  # List to store deltas of this layer
        if next is None:
            assert expected.shape[0] == len(self.perceptrons), "Expected vector must be in the same dimension of the output vector"
            # Output layer: match each perceptron with its expected label
            for perceptron, y in zip(self.perceptrons, expected):
                deltas.append(perceptron.get_deltas(expected=y, clip_size= clip_size))
        else:
            for perceptron in self.perceptrons:
                deltas.append(perceptron.get_deltas(next=next, clip_size= clip_size))

        self.deltas = np.array(deltas)
        return np.array(deltas)


# Defining a Neural Network
class NeuralNetwork:
    def __init__(self, layers: list[Layer], verbose: bool = False):
        self.layers = layers
        self.out: Vector
        self.verbose = verbose

    def forward(self, input_data: Vector, clip_size: float = 500) -> Vector:
        current_input = input_data
        for layer in self.layers:
            layer.inputs = current_input
            if self.verbose:
                print(f"Forward Propagation - Layer {layer.num}: input = {input_data}")
            current_input = layer.get_out(clip_size= clip_size)
        self.out = current_input
        if self.verbose:
            print(f"Neural Network Output: {self.out}")
        return self.out
    
    def backward(self, expected: Vector, clip_size: float = 500):
        next_layer = None  # No next layer yet (start from output)

        if self.verbose:
            print(f"Starting Backpropagation with expected output: {expected}")
            
        for layer in reversed(self.layers):  # Process layers from output back to input
            if next_layer is None:
                # Output layer: use the expected outputs
                if self.verbose:
                    print(f"Layer {layer.num} (Output Layer) - Computing deltas...")
                layer.get_deltas(expected=expected, clip_size= clip_size)
            else:
                # Hidden layer: use next layer's perceptrons
                if self.verbose:
                    print(f"Layer {layer.num} (Hidden Layer) - Computing deltas from next layer...")
                layer.get_deltas(next=next_layer, clip_size= clip_size)

            # Debugging output for deltas in the current layer
            if self.verbose:
                deltas = np.array([p.delta for p in layer.perceptrons])
                print(f"Layer {layer.num} - Deltas: {deltas}")

            # Move one layer backward
            next_layer = layer

        if self.verbose:
            print("Backpropagation complete.")

 
                

