from Trainers import *

data = Dataset("Data/Xor.csv", )

#test: 
"""
seeds   -   min loss   -  epochs  - learning rate - verbose step
668792  -   0.001936   -  10000   -       2       -     1000
"""
layer1_weights = initialize_weights(2, 2, seed= 668792)   
output_weights = initialize_weights(2, 1, seed= 668792)

# Create layers
layer1 = Layer(weights=layer1_weights, num_of_perceptrons=2, activation_method="relu", num="1")
output = Layer(weights=output_weights, num_of_perceptrons=1, activation_method="relu", num="2")

# Create the Neural Network
nn = NeuralNetwork(layers=[layer1, output])

trainer = Trainer(data, nn, learning_rate= 1)
trainer.train(epochs= 10000, verbose= True, verbose_step=1000)

err = 0
# Test the trained network
print("\nTesting the trained network:")
for i in range(len(data.test_data[0])):
    output = nn.forward(data.test_data[0][i])
    if (loss(output, data.test_data[1][i])) > 0.25: err += 1
    print(f"Input: {data.test_data[0][i]} -> Predicted: {output}, Expected: {data.test_data[1][i]}")

print(f"Precision: {(1 - err/len(data.test_data[0])) * 100}%")

print("---Welcome to simple Linear Regression NN!!---")
while True:
    x1 = float(input("x1: "))
    x2 = float(input("x2: "))
    out = nn.forward(np.array([x1, x2]))
    print(out)