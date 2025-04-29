import numpy as np
from Neural_Network import *

#good seeds: 72026, 668792

# Create a simple dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (4 samples, 2 features)
y = np.array([0, 1, 1, 0])  # Expected output (XOR logic)

# Define the architecture of the neural network
# 2 input neurons, 2 neurons in the hidden layer, 1 output neuron
layer1_weights = initialize_weights(2, 2, verbose= True)  
layer2_weights = initialize_weights(2, 2, verbose= True)
layer3_weights = initialize_weights(2, 2, verbose= True)  
output_weights = initialize_weights(2, 1, verbose= True)

# Create layers
layer1 = Layer(weights=layer1_weights, num_of_perceptrons=2, activation_method="sigmoid", num="1")
layer2 = Layer(weights=layer2_weights, num_of_perceptrons=2, activation_method="sigmoid", num="2")
layer3 = Layer(weights=layer3_weights, num_of_perceptrons=2, activation_method="sigmoid", num="3")
output = Layer(weights=output_weights, num_of_perceptrons=1, activation_method="sigmoid", num="4")

# Create the Neural Network
nn = NeuralNetwork(layers=[layer1, layer2, layer3, output])

# Training loop
nn.train(X, y, 10000, verbose= True)

# Test the trained network
print("\nTesting the trained network:")
for i in range(len(X)):
    output = nn.forward(X[i])
    print(f"Input: {X[i]} -> Predicted: {output}, Expected: {y[i]}")

print("---Welcome to simple Xor NN!!---")
while True:
    x1 = float(input("x1: "))
    x2 = float(input("x2: "))
    out = nn.forward(np.array([x1, x2]))
    if out >= 0.5:
        print("activated")
    else:
        print("not activated")