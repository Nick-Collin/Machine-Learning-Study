"""
Utility functions:
    weight_initialization,
    activation functions,
    loss calculations.
"""

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
def activation(x: float, mode: str = "", derivative: bool = False, clip_size: float = 500) -> float:
    assert isinstance(x, float)

    if derivative:
        match mode:
            case "sigmoid":
                s = 1 / (1 + np.exp(-np.clip(x, -clip_size, clip_size)))
                return s * (1 - s)
            
            case "tanh":
                t = np.tanh(x)
                return 1 - t**2
            
            case "relu":
                return 1.0 if x > 0 else 0.0
            
            case "leaky_relu":
                return 1.0 if x >= 0 else 0.01
            
            case _:
                return 1.0
    else:
        match mode:
            case "sigmoid":
                return 1 / (1 + np.exp(-np.clip(x, -clip_size, clip_size)))
            
            case "tanh":
                return np.tanh(x)
            
            case "relu":
                return max(0.0, x)
            
            case "leaky_relu":
                return x if x > 0 else 0.01 * x
            
            case _:
                return x

# Cost function (every element of a layer)
def cost(x: Vector, y: Vector, mode: str = "", derivative: bool = False) -> Vector:
    if isinstance(y, np.ndarray):
        # Handle vector input
        assert x.shape == y.shape, f"x({x.shape}) and y({y.shape}) must have the same shape"
    

    # Handle vector input case
    if derivative:
        match mode:
            case "cross_entropy":
                return (x - y)
            case _:
                # Default case L`(x,y) = (x - y) * 2
                return 2 * (x - y)
    else:
        match mode:
            case "cross_entropy":
                return -y * np.log(x) - (1 - y) * np.log(1 - x)
            
            case _:
                # Default case L(x,y) = (x - y) ** 2
                return np.square(np.subtract(x, y))  # This handles vector subtraction

# Loss function (sum of all costs)
def loss(x: Vector, y: Vector, mode: str = ""):
    assert isinstance (x, np.ndarray), "x must be a vector"
    assert isinstance (y, np.ndarray), "y must be a vector"
    assert x.ndim == y.ndim , f"x({x.ndim}) and y({y.ndim}) are supposed to be the same dimension"

    # TODO Implement more loss methods
    match mode:
        case _:
            # Default case L(x,y) = (x - y) ** 2
            return np.sum(cost(x, y))