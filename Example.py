from ninc.Util import logging, np, norm_input, denorm_output
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, PolyakMomentum
from ninc.DataHandeling import Dataset, Ratio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define dataset path and split ratio
    dataset_path = "Data/LinearRegression.csv"
    split_ratio = Ratio(training=0.7, validation=0.2, testing=0.1)

    # Load and split dataset
    dataset = Dataset(path=dataset_path, split_ratio=split_ratio)
    dataset.load()
    dataset.split(label_column="out")
    dataset.normalize(normalize_labels= True)

    # Define neural network architecture
    layers = [ # Match input features (2)
        Layer(num_of_units=1, activation_mode="linear", initialization_mode= "He")
    ]
    nn = NeuralNetwork(layers=layers, loss_func="MSE", dataset=dataset)

    # Define optimizer and trainer
    optimizer = PolyakMomentum()
    trainer = Trainer(nn=nn, optimizer=optimizer, epochs=1000, learning_rate=0.01, checkpoint=1)

    # Train the model
    trainer.train(batched=True, batch_size=32, shuffle=True)

    # Evaluate the model (example)
    print("Training complete. Evaluate the model as needed.")

    testing_mode(nn, dataset)

def testing_mode(nn: NeuralNetwork, dataset):
    while True:
        try:
            x1 = int(input("x1: "))
            x2 = int(input("x2: "))
            inputs = np.array([x1, x2])

        except:
            print("Couldn't resolve inputs, bailing out!")
            exit()

        out = nn.propagate_forward(norm_input(inputs, dataset))
        out = denorm_output(out, dataset)
        print(f"Out: {out}")

if __name__ == "__main__":
    main()