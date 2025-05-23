from neurolite.Neural_Network import NeuralNetwork
from neurolite.DataHandling import Dataset, Ratio
from neurolite.Evaluators import BinaryClassificationEvaluator

dt = Dataset("Data/diabetes.csv", Ratio(0.1, 0.1, 0.8))
dt.load()
dt.split(label_column="Outcome", shuffle=True)


nn = NeuralNetwork.load("SavedModels/diabetes_P86%.npz", dt)

eval = BinaryClassificationEvaluator.evaluate(nn, dt.testing)
print(f"{eval}")