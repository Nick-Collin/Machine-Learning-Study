import logging
import sys
logging.basicConfig(filename="log.txt", level=logging.INFO, filemode="w")
sys.stdout = open("log.txt", "a")
sys.stderr = open("log.txt", "a")

from ninc.DataHandling import Dataset, Ratio
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, Adam

# Load and prepare data
dataset = Dataset(path="Data/diabetes.csv", split_ratio=Ratio(0.7, 0.2, 0.1))
dataset.load()
dataset.split(label_column="Outcome", one_hot=True)
dataset.normalize()

# Define network architecture
layers = [
    Layer(num_of_units=16, activation_mode="relu", initialization_mode="He"),
    Layer(num_of_units=2, activation_mode="sigmoid", initialization_mode="Xavier")
]

# Initialize network and trainer
nn = NeuralNetwork(layers=layers, loss_func="CrossEntropy", dataset=dataset)
optimizer = Adam()
trainer = Trainer(nn=nn, optimizer=optimizer, epochs=100, learning_rate=0.001)

# Train and evaluate
trainer.train(batched=True, batch_size=32, shuffle=True)
trainer.test()
trainer.calculate_precision(dataset_type="testing")
nn.save("diabetes.npz")
