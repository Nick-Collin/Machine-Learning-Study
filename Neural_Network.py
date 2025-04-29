import numpy as np


# Defining vector and matrix types
type Vector = np.ndarray
type Matrix = np.ndarray

# Xavier weight initialization
def initialize_weights(input_size: int, num_perceptrons: int, seed: int = np.random.randint(0, 999999), verbose: bool = False) -> Matrix:
    if verbose: print(f"weights seed: {seed}")
    np.random.seed(seed=seed)
    return np.random.randn(num_perceptrons, input_size + 1) * np.sqrt(2. / input_size)

# Defining mathematical functions
# Activation function
def activation(x: float, mode: str = "", derivative: bool = False, clip_size: int = 500) -> float:
    assert isinstance(x, float)

    # TODO Implement more activation methods
    if derivative:
        match mode:
            case "sigmoid":
                s = 1 / (1 + np.exp(-np.clip(x, -clip_size, clip_size)))
                return s * (1 - s)
            
            case "tanh":
                t = np.tanh(np.clip(x, -clip_size, clip_size))
                return 1 - t**2
            
            case "relu":
                return 1.0 if np.clip(x, -clip_size, clip_size) > 0 else 0.0
            
            case "leaky_relu":
                return 1.0 if np.clip(x, -clip_size, clip_size) > 0 else 0.01

            case _:
                # Default derivative (linear): dy/dx = 1
                return 1.0
    else:
        match mode:
            case "sigmoid":
                return 1 / (1 + np.exp(-np.clip(x, -clip_size, clip_size)))
            
            case "tanh":
                return np.tanh(np.clip(x, -clip_size, clip_size))
            
            case "relu":
                return max(0.0, np.clip(x, -clip_size, clip_size))
            
            case "leaky_relu":
                return np.clip(x, -clip_size, clip_size) if np.clip(x, -clip_size, clip_size) > 0 else 0.01 * np.clip(x, -clip_size, clip_size)

            case _:
                # Default activation: identity (f(x) = x)
                return np.clip(x, -clip_size, clip_size)

# Cost function (one element)
def cost(x: float, y: float, mode: str = "", derivative: bool = False) -> float:
    if isinstance(y, np.ndarray):
        # Handle vector input
        assert x.shape == y.shape, f"x({x.shape}) and y({y.shape}) must have the same shape"
    
    assert isinstance(x, (float, np.generic)), "x must be a float or a numpy array"
    assert isinstance(y, (float, np.generic)), "y must be a float or a numpy array"

    if derivative:
        match mode:
            case _:
                # Default case L`(x,y) = (x - y) * 2
                return 2 * (x - y)
    else:
        match mode:
            case _:
                # Default case L(x,y) = (x - y) ** 2
                return np.square(np.subtract(x, y))  # This handles vector subtraction

# Cost function (every element of a layer)
def costvec(x: Vector, y: Vector, mode: str = "") -> Vector:
    assert isinstance(x, np.ndarray), "x must be a vector"
    assert isinstance(y, np.ndarray), "y must be a vector"
    assert x.ndim == y.ndim, f"x({x.ndim}) and y({y.ndim}) must have the same dimension"

    # Handle vector input case
    match mode:
        case _:
            # Default case L(x,y) = (x - y) ** 2
            return np.square(np.subtract(x, y))

# Loss function (sum of all costs)
def loss(x: Vector, y: Vector, mode: str = ""):
    assert isinstance (x, np.ndarray), "x must be a vector"
    assert isinstance (y, np.ndarray), "y must be a vector"
    assert x.ndim == y.ndim , f"x({x.ndim}) and y({y.ndim}) are supposed to be the same dimension"

    # TODO Implement more loss methods
    match mode:
        case _:
            # Default case L(x,y) = (x - y) ** 2
            return np.sum(costvec(x, y))

# Defining a perceptron
class Perceptron:
    def __init__(self, layer: str = "not provided", num: str = "not provided", activation_method: str = "Linear"):
        # Debug only
        self.layer = layer
        self.num = int(num)
        self.activation_method = activation_method

    def activate(self, inputs: Vector, weights: Vector, bias: float, verbose: bool = False, clip_size: int = 500) -> float:
        assert isinstance(inputs, np.ndarray), f"expected a input vector at layer: {self.layer}, perceptron: {self.num}"
        assert isinstance(weights, np.ndarray), f"expected a weight vector at layer: {self.layer}, perceptron: {self.num}"
        assert inputs.ndim == 1, "Inputs must be a vector"
        assert weights.ndim == 1, "Weights must be a vector"
        assert inputs.shape == weights.shape, "Inputs and weights mismatch shapes"

        self.weights = weights
        self.z = np.dot(inputs, weights) + bias
        if verbose:
            print(f"Perceptron {self.num} in Layer {self.layer}: inputs({inputs}), weights({weights}), bias({bias})")
            print(f"Sum (z) = {self.z}, dot product = {np.dot(inputs, weights)}")

        self.a = activation(self.z, mode=self.activation_method, clip_size= clip_size)
        if verbose:
            print(f"Activated output (a) = {self.a}")
        return self.a

    def get_deltas(self, next: "Layer" = None, expected: float = None, verbose: bool = False, clip_size: float = 500) -> float:
        if next is None:
            # If there's no "next", we are at the output layer
            # Delta = cost derivative * activation derivative
            delta = cost(self.a, expected, derivative=True) * activation(self.z, mode=self.activation_method, derivative=True, clip_size= clip_size)
            if verbose:
                print(f"Perceptron {self.num} in Layer {self.layer} (Output Layer) - Delta: {delta}")
        else:
            assert next.deltas.ndim == next.weights.T[self.num].ndim, f"The next Layer's deltas vector must be the same dimension as the weight's vec associated with it"
            # If we have a next layer, we're in a hidden layer
            # Delta = (sum of next deltas * weights) * activation derivative
            delta = np.dot(next.deltas, next.weights.T[self.num]) * activation(self.z, mode=self.activation_method, derivative=True)
            if verbose:
                print(f"Perceptron {self.num} in Layer {self.layer} (Hidden Layer) - Delta: {delta}")
        
        # Save delta for future use
        self.delta = delta
        return delta


# Defining a layer
class Layer:
    def __init__(self, weights: Matrix, num_of_perceptrons: int, activation_method: str = "", num: str = "not provided", inputs: Vector = None):
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

        for k in range(num_of_perceptrons):
            h = Perceptron(self.num, f"{k}", activation_method=self.activation_method)
            self.perceptrons.append(h)

    def get_out(self, verbose: bool = False, clip_size: float = 500) -> Vector:
        out = []
        k = 0
        for perceptron in self.perceptrons:
            # We limit the weight vector to not include the last scalar because it represents the bias
            # Important note: here is crucial to update the perceptron inputs, because elsewhere it would keep with the inputs set at init time before the forward propagation
            a = perceptron.activate(inputs=self.inputs, weights=self.weights[k][:-1], bias=self.weights[k][-1], verbose=verbose, clip_size= clip_size)
            out.append(a)
            k += 1

            # Debug
            if verbose:
                print(f"Layer {self.num} - Perceptron {perceptron.num}: activation = {a}, current output = {out}")

        if verbose:
            print(f"Layer {self.num} final output: {out}")
        return np.array(out)

    def get_deltas(self, next: "Layer" = None, expected: Vector = None, verbose: bool = False, clip_size: float = 500) -> Vector:
        deltas = []  # List to store deltas of this layer

        if next is None:
            assert expected.shape[0] == len(self.perceptrons), "Expected vector must be in the same dimension of the output vector"
            # Output layer: match each perceptron with its expected label
            for perceptron, y in zip(self.perceptrons, expected):
                deltas.append(perceptron.get_deltas(expected=y, verbose=verbose, clip_size= clip_size))
        else:
            for perceptron in self.perceptrons:
                deltas.append(perceptron.get_deltas(next=next, verbose=verbose, clip_size= clip_size))

        self.deltas = np.array(deltas)
        return np.array(deltas)


# Defining a Neural Network
class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.out: Vector

    def forward(self, input_data: Vector, verbose: bool = False, clip_size: float = 500) -> Vector:
        current_input = input_data
        for layer in self.layers:
            layer.inputs = current_input
            if verbose:
                print(f"Forward Propagation - Layer {layer.num}: input = {input_data}")
            current_input = layer.get_out(verbose, clip_size= clip_size)
        self.out = current_input
        if verbose:
            print(f"Neural Network Output: {self.out}")
        return self.out
    
    def backward(self, expected: Vector, verbose: bool = False, clip_size: float = 500):
        next_layer = None  # No next layer yet (start from output)

        if verbose:
            print(f"Starting Backpropagation with expected output: {expected}")
            
        for layer in reversed(self.layers):  # Process layers from output back to input
            if next_layer is None:
                # Output layer: use the expected outputs
                if verbose:
                    print(f"Layer {layer.num} (Output Layer) - Computing deltas...")
                layer.get_deltas(expected=expected, verbose=verbose, clip_size= clip_size)
            else:
                # Hidden layer: use next layer's perceptrons
                if verbose:
                    print(f"Layer {layer.num} (Hidden Layer) - Computing deltas from next layer...")
                layer.get_deltas(next=next_layer, verbose=verbose, clip_size= clip_size)

            # Debugging output for deltas in the current layer
            if verbose:
                deltas = np.array([p.delta for p in layer.perceptrons])
                print(f"Layer {layer.num} - Deltas: {deltas}")

            # Move one layer backward
            next_layer = layer

        if verbose:
            print("Backpropagation complete.")

    def train(self, input_data: Matrix, expected: Matrix, epochs: int, learning_rate: int = 1, verbose: bool = False, verbose_step: int = 1000, clip_size: float = 500, verboseplus: bool= False, alpha_update_factor: float = 0.5):
        
        last = 0
        for i in range(len(input_data)):
            inputs = input_data[i]
            expected_output = np.array([expected[i]])

            # Forward pass
            self.forward(inputs, clip_size= clip_size, verbose= verboseplus)

            # Compute loss for the firt iteration
            loss_value = loss(self.out, expected_output)
            last += loss_value

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(input_data)):
                inputs = input_data[i]
                expected_output = np.array([expected[i]])

                # Forward pass
                self.forward(inputs, clip_size= clip_size, verbose= verboseplus)

                # Compute loss
                loss_value = loss(self.out, expected_output)
                total_loss += loss_value

                # Backward pass (backpropagation)
                self.backward(expected_output, clip_size= clip_size, verbose= verboseplus)

                # Update weights using the deltas from backpropagation
                # Simple gradient descent update
                for layer in self.layers:
                    for perceptron in layer.perceptrons:
                        perceptron.weights -= learning_rate * perceptron.delta * layer.inputs

            #Update learning rate
            if total_loss > last:
                learning_rate = learning_rate * alpha_update_factor
            last = total_loss

            # Print the loss 
            if epoch % verbose_step == 0 and verbose:
                print(f"Epoch {epoch}/{epochs} - Loss: {total_loss} - Alpha: {learning_rate}")

                

