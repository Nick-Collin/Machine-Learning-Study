import logging
import numpy as np
from neurolite.DataHandling import Dataset, Ratio
from neurolite.Neural_Network import NeuralNetwork, Layer
from neurolite.Trainer import Trainer
from neurolite.Evaluators import BinaryClassificationEvaluator
from neurolite.Optimizers import Adam, SGD, PolyakMomentum, RMSProp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode= 'w',
                    filename= "log.txt")

# Initialize dataset
dataset = Dataset(
    path="Data/diabetes.csv",
    split_ratio=Ratio(training=0.7, validation=0.2, testing=0.1)
)

# Load data
dataset.load()

# Handle missing data (0s represent missing values in some features)
medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
dataset.handle_missing_data(
    columns=medical_features,
    strategy="mean",
    placeholder=0
)

# Split dataset
dataset.split(
    label_column="Outcome",
    one_hot=False,  # Binary classification (0/1 labels)
    shuffle=True
)

# Normalize features
dataset.normalize(mode="standardization")

# Split batches
dataset.split_batches()

# Define network architecture
layers = [
    Layer(32, activation_mode="relu", initialization_mode="He"),
    Layer(8, activation_mode= "relu", initialization_mode= "He"),
    Layer(1, activation_mode="sigmoid", initialization_mode="Xavier")
]

# Initialize neural network
nn = NeuralNetwork(
    dataset=dataset,
    layers=layers,
    loss_func="CrossEntropy"  # For binary classification
)

# Set up trainer with Adam optimizer
optimizer = Adam()
trainer = Trainer(
    nn=nn,
    optimizer=optimizer,
    epochs=1000,
    learning_rate=0.001,
    checkpoint=10,
    early_stopping=True,
    patience=100,
)

# Train the model
logging.info("Starting training...")
trainer.train(batched=True, batch_size=32, shuffle=True)

# Evaluate
eval = BinaryClassificationEvaluator.evaluate(model= nn, dataset=nn.dataset.testing)
logging.info("\nFinal Evaluation:")
logging.info(f"\nTesting Precision: {eval["precision"]}\nTesting Recal: {eval["recall"]}\nTesting F1 Score: {eval["f1"]}")

print(f"Testing Precision: {eval["precision"]}\n Testing Recal: {eval["recall"]}\n Testing F1 Score: {eval["f1"]}")

nn.save(f"SavedModels/diabetes_P{round(eval["precision"]*100)}%.npz")
