"""
Training routines
"""

from Neural_Network import * 

class Dataset:
    def __init__(self, 
                 path: str = "", 
                 val_ratio:float = 0.1, 
                 train_ratio: float = 0.8, 
                 verbose: bool = False):
        
        self.val_ratio, self.train_ratio = val_ratio, train_ratio
        self.data, self.labels = self.load_data(path)
        if verbose: print(f"Loaded:\n Data:\n {self.data}\n Labels:\n {self.labels}")
        self.train_data, self.validation_data, self.test_data = self.split_data()
        if verbose: print(f"Splited:\n Training:\n {self.train_data}\n Testing:\n {self.test_data}")

    def load_data(self, path):
        # Load from CSV, image folder, etc.
        
        if path.endswith(".csv"):
            from csv import reader
            data = []
            labels = []

            with open(path, newline='') as file:
                csv_reader = reader(file)

                header = next(csv_reader)  # First line: column names
                outsize = sum(1 for col in header if col.startswith("out"))

                for row in csv_reader:
                    # Convert to floats
                    row = [float(x) for x in row]

                    # Features are all but last `outsize` elements
                    features = row[:-outsize]
                    label = row[-outsize:]  # now a list of outputs

                    data.append(features)
                    labels.append(label)
            return np.array(data), np.array(labels)

        elif path.endswith(".json"):
            from json import load
            with open(path, 'r') as file:
                dataset = load(file)

            data = [item['features'] for item in dataset]
            labels = [item['label'] for item in dataset]
            return np.array(data), np.array(labels)

        else: print("Invalid path extension")

    def split_data(self):
        # Shuffle the data
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        data = self.data[indices]
        labels = self.labels[indices]

        # Calculate split points
        total = len(self.data)
        train_end = int(total * self.train_ratio)
        val_end = int(total * (self.train_ratio + self.val_ratio))

        # Split the data
        train_data, train_labels = self.data[:train_end], self.labels[:train_end]
        val_data, val_labels = self.data[train_end:val_end], self.labels[train_end:val_end]
        test_data, test_labels = self.data[val_end:], self.labels[val_end:]

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

class Trainer:
    def __init__(self, 
                 dataset: Dataset, 
                 nn: NeuralNetwork, 
                 learning_rate: int = 1):
        
        self.learning_rate = learning_rate
        self.nn = nn
        self.dataset = dataset

    def train(self, 
              epochs:int = 50, 
              verbose: bool = False, 
              verbose_step: int = 1000, 
              clip_size: float = 500, 
              loss_mode: str = "MSE"):
            
            last = 0
            for i in range(len(self.dataset.train_data[0])):
                inputs = self.dataset.train_data[0][i]
                expected_outputs = self.dataset.train_data[1][i]

                # Forward pass
                self.nn.forward(inputs, clip_size= clip_size)

                # Compute loss for the firt iteration
                loss_value = loss(self.nn.out, expected_outputs, mode= loss_mode)
                last += loss_value

            for epoch in range(epochs):
                total_loss = 0
                for i in range(len(self.dataset.train_data[0])):
                    inputs = self.dataset.train_data[0][i]
                    expected_outputs = self.dataset.train_data[1][i]

                    # Forward pass
                    self.nn.forward(inputs, clip_size= clip_size)

                    # Compute loss
                    loss_value = loss(self.nn.out, expected_outputs, loss_mode)
                    total_loss += loss_value

                    # Backward pass (backpropagation)
                    self.nn.backward(expected_outputs, clip_size= clip_size)

                    # Update weights using the deltas from backpropagation
                    # Simple gradient descent update
                    for layer in self.nn.layers:
                        for perceptron in layer.perceptrons:
                            #For the weights itself
                            perceptron.weights -= self.learning_rate * perceptron.delta * layer.inputs
                            #For the bias therm
                            perceptron.bias -= self.learning_rate * perceptron.delta 


                # Print the loss 
                if epoch % verbose_step == 0 and verbose:
                    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss} - Alpha: {self.learning_rate}")

class SDG_Dinamic_Alpha(Trainer):
    '''
    Updates alpha by multiplying it by a factor, if L' > L
    '''

    def __init__(self, 
                 dataset: Dataset, 
                 nn: NeuralNetwork, 
                 learning_rate: int = 1, 
                 alpha_update_factor: float = 0.5):
        
        super().__init__(dataset=dataset, nn=nn, learning_rate= learning_rate)
        self.alpha_update_factor = alpha_update_factor


    def train(self, 
              epochs:int = 50, 
              verbose: bool = False, 
              verbose_step: int = 1000, 
              clip_size: float = 500, 
              loss_mode: str = "MSE"):
            
            last = 0
            for i in range(len(self.dataset.train_data[0])):
                inputs = self.dataset.train_data[0][i]
                expected_outputs = self.dataset.train_data[1][i]

                # Forward pass
                self.nn.forward(inputs, clip_size= clip_size)

                # Compute loss for the firt iteration
                loss_value = loss(self.nn.out, expected_outputs, mode= loss_mode)
                last += loss_value

            for epoch in range(epochs):
                total_loss = 0
                for i in range(len(self.dataset.train_data[0])):
                    inputs = self.dataset.train_data[0][i]
                    expected_outputs = self.dataset.train_data[1][i]

                    # Forward pass
                    self.nn.forward(inputs, clip_size= clip_size)

                    # Compute loss
                    loss_value = loss(self.nn.out, expected_outputs, mode= loss_mode)
                    total_loss += loss_value

                    # Backward pass (backpropagation)
                    self.nn.backward(expected_outputs, clip_size= clip_size)

                    # Update weights using the deltas from backpropagation
                    # Simple gradient descent update
                    for layer in self.nn.layers:
                        for perceptron in layer.perceptrons:
                            #For the weights itself
                            perceptron.weights -= self.learning_rate * perceptron.delta * layer.inputs
                            #For the bias therm
                            perceptron.bias -= self.learning_rate * perceptron.delta 

                #Update learning rate
                if total_loss > last:
                    self.learning_rate = self.learning_rate * self.alpha_update_factor
                last = total_loss

                # Print the loss 
                if epoch % verbose_step == 0 and verbose:
                    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss} - Alpha: {self.learning_rate}")



