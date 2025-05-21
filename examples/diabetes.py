from ninc.Util import logging, np, norm_input, denorm_output
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, Adam
from ninc.DataHandling import Dataset, Ratio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define dataset path and split ratio
    dataset_path = "Data/diabetes.csv"
    split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

    # Load and prepare dataset
    dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
    dataset.load()
    dataset.split(label_column="Outcome") 
    dataset.normalize(normalize_labels=True)

    # Define neural network architecture
    input_dim = dataset.training.features.shape[1]
    layers = [
        Layer(num_of_units=32, activation_mode="relu", initialization_mode="He"),
        Layer(num_of_units=16, activation_mode="relu", initialization_mode="He"),
        Layer(num_of_units=1, activation_mode="linear", initialization_mode="He")
    ]

    nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)

    # Define optimizer and trainer
    optimizer = Adam()
    trainer = Trainer(nn=nn, optimizer=optimizer, epochs=200, learning_rate=0.001, checkpoint=1)

    # Train the model
    trainer.train(batched=True, batch_size=32, shuffle=True)

    # Evaluate on test set
    trainer.test()

if __name__ == "__main__":
    main()
