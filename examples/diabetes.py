import logging
import numpy as np
from ninc.DataHandling import Dataset, Ratio
from ninc.Neural_Network import NeuralNetwork, Layer
from ninc.Trainers import Trainer, Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
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
    
    # Define network architecture
    layers = [
        Layer(64, activation_mode="relu", initialization_mode="He"),
        Layer(32, activation_mode="relu", initialization_mode="He"),
        Layer(1, activation_mode="softmax", initialization_mode="Xavier")  # Binary output
    ]
    
    # Initialize neural network
    nn = NeuralNetwork(
        dataset=dataset,
        layers=layers,
        loss_func="CrossEntropy"  # For binary classification
    )
    
    # Set up trainer with Adam optimizer
    optimizer = Adam(beta1=0.9, beta2=0.999, epsilon=1e-8)
    trainer = Trainer(
        nn=nn,
        optimizer=optimizer,
        epochs=200,
        learning_rate=0.001,
        checkpoint=10,
        early_stopping=True,
        patience=15,
        min_delta=0.001
    )
    
    # Train the model
    logging.info("Starting training...")
    trainer.train(batched=True, batch_size=32, shuffle=True)
    
    # Evaluate
    logging.info("\nFinal Evaluation:")
    trainer.test()
    logging.info("Testing Precision:")
    trainer.calculate_precision(dataset_type="testing")
    
    # Optional: Validate on training and validation sets
    logging.info("\nTraining Precision:")
    trainer.calculate_precision(dataset_type="training")
    logging.info("Validation Precision:")
    trainer.calculate_precision(dataset_type="validation")

    nn.save("SavedModels/diabetes.npz")

if __name__ == "__main__":
    main()