import pandas as pd
from Util import np

def to_data(df: pd.DataFrame,
            label_column: str = "label") -> "Data":
    features = df.drop(columns=[label_column]).to_numpy()
    labels = df[label_column].to_numpy()
    return Data(features, labels)

class Data:
    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray):
        
        self.features: np.ndarray = features
        self.labels: np.ndarray = labels

class Ratio:
    def __init__(self,
                 training: float,
                 validation: float,
                 testing: float):
        
        self.training: float = training
        self.validation: float = validation
        self.testing: float = testing

        if not np.isclose(self.training + self.validation + self.testing, 1.0):
            raise ValueError("Sum of ratios must equal 1.")

class Dataset:
    def __init__(self,
                 path: str,
                 split_ratio: Ratio):
        
        self.path: str = path
        self.split_ratio: Ratio = split_ratio
        self.batches: list[Data] = []
        self.data: pd.DataFrame
           
        self.training: Data
        self.validation: Data
        self.testing: Data
    
    def load(self):
        try:
            if self.path.endswith(".csv"):
                self.data = pd.read_csv(self.path)
            elif self.path.endswith(".xlsx"):
                self.data = pd.read_excel(self.path)
            else:
                raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the dataset: {e}")

    def split(self, 
            shuffle: bool = True,
            label_column: str = "label"):
        if not hasattr(self, "data"):
            raise RuntimeError("You must call `load()` before `split()`.")

        df = self.data

        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        total_samples = len(df)
        train_end = int(total_samples * self.split_ratio.training)
        val_end = train_end + int(total_samples * self.split_ratio.validation)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        self.training = to_data(train_df, label_column)
        self.validation = to_data(val_df, label_column)
        self.testing = to_data(test_df, label_column)

    def split_batches(self, batch_size: int, shuffle: bool = True) -> list[Data]:
        features = self.training.features
        labels = self.training.labels
        num_samples = features.shape[0]

        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            features = features[indices]
            labels = labels[indices]

        self.batches = [
            Data(
                features=features[i:i+batch_size],
                labels=labels[i:i+batch_size]
            )
            for i in range(0, num_samples, batch_size)
        ]
        return self.batches