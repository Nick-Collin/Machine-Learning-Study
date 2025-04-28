import numpy as np
from Neural_Network import *

# Create a simple dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (4 samples, 2 features)
y = np.array([0, 1, 1, 0])  # Expected output (XOR logic)

# Define the architecture of the neural network
# 2 input neurons, 2 neurons in the hidden layer, 1 output neuron
layer1_weights = initialize_weights(2, 2)  # 2 perceptrons, each with 2 input values + 1 bias
layer2_weights = initialize_weights(2, 1)  # 1 output perceptron, 2 inputs + 1 bias

# Create layers
layer1 = Layer(weights=layer1_weights, num_of_perceptrons=2, activation_method="sigmoid", num="1")
layer2 = Layer(weights=layer2_weights, num_of_perceptrons=1, activation_method="sigmoid", num="2")

# Create the Neural Network
nn = NeuralNetwork(layers=[layer1, layer2])

# Training loop
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        input_data = X[i]
        expected_output = np.array([y[i]])

        # Forward pass
        nn.forward(input_data)

        # Compute loss
        loss_value = loss(nn.out, expected_output)
        total_loss += loss_value

        # Backward pass (backpropagation)
        nn.backward(expected_output)

        # Update weights using the deltas from backpropagation
        # Simple gradient descent update
        for layer in nn.layers:
            for perceptron in layer.perceptrons:
                perceptron.weights -= learning_rate * perceptron.delta * input_data

    # Print the loss every 1000 epochs for debugging
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss}")

# Test the trained network
print("\nTesting the trained network:")
for i in range(len(X)):
    output = nn.forward(X[i])
    print(f"Input: {X[i]} -> Predicted: {output}, Expected: {y[i]}")
