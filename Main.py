from Util import logging
from Neural_Network import NeuralNetwork, Layer
from Trainers import Trainer, SGD, PolyakMomentum
from DataHandeling import Dataset, Ratio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define dataset path and split ratio
    dataset_path = "Data/Xor.csv"
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
    trainer = Trainer(nn=nn, optimizer=optimizer, epochs=100000, learning_rate=0.01, checkpoint=100)

    # Train the model
    trainer.train(batched=True, batch_size=32, shuffle=True)

    # Evaluate the model (example)
    print("Training complete. Evaluate the model as needed.")

if __name__ == "__main__":
    main()

