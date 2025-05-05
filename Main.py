"""
Main module for the Machine Learning project.

This module serves as the entry point for the project. It is responsible for initializing and orchestrating the various components of the project, such as loading datasets, training models, and evaluating results.

Functions:
    main(): Entry point for the project. Initializes datasets, trains models, and evaluates results.
"""

from Util import Dataset, Ratio
from Neural_Network import NeuralNetwork, Layer
from Trainers import Trainer, SGD



def main():
    """
    Entry point for the Machine Learning project.

    This function initializes datasets, trains models, and evaluates results.
    """
    # Define dataset path and split ratio
    dataset_path = "Data/LinearRegression.csv"
    split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

    # Load and split dataset
    dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
    dataset.load()
    dataset.split(label_column="out")

    # Define neural network architecture
    layers = [
        Layer(num_of_units=2, activation_mode="sigmoid"),  # Match input features (2)
        Layer(num_of_units=1, activation_mode="sigmoid")
    ]
    nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)

    # Define optimizer and trainer
    optimizer = SGD()
    trainer = Trainer(nn=nn, optimizer=optimizer, epochs=10, learning_rate=0.01)

    # Train the model
    trainer.train(batched=True, batch_size=32, shuffle=True)

    # Evaluate the model (example)
    print("Training complete. Evaluate the model as needed.")

if __name__ == "__main__":
    main()

