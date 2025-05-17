import pandas as pd
from ninc.Util import np

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
    
    def normalize(self, 
                  mode: str = "standardization", 
                  normalize_labels: bool = False):
            
            if not (hasattr(self, 'training') and hasattr(self, 'validation') and hasattr(self, 'testing')):
                raise AttributeError("Data must be split before normalization!")

            # Save for normalization in deploy
            self.norm_mode = mode

            # Compute normalization parameters from training data
            train_features = self.training.features
            match mode:
                case "standardization":
                    self.feature_mean = np.mean(train_features, axis=0)
                    self.feature_std = np.std(train_features, axis=0)
                    self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero
                case "minmax":
                    self.feature_min = np.min(train_features, axis=0)
                    self.feature_max = np.max(train_features, axis=0)
                    self.feature_range = self.feature_max - self.feature_min
                    self.feature_range[self.feature_range == 0] = 1.0
                case _:
                    raise ValueError(f"Unsupported mode: {mode}")

            # Apply normalization to all datasets
            for data in [self.training, self.validation, self.testing]:
                if mode == "standardization":
                    data.features = (data.features - self.feature_mean) / self.feature_std
                elif mode == "minmax":
                    data.features = (data.features - self.feature_min) / self.feature_range

            # Normalize labels if required
            if normalize_labels:
                train_labels = self.training.labels
                match mode:
                    case "standardization":
                        self.label_mean = np.mean(train_labels)
                        self.label_std = np.std(train_labels)
                        self.label_std = self.label_std if self.label_std != 0 else 1.0
                        for data in [self.training, self.validation, self.testing]:
                            data.labels = (data.labels - self.label_mean) / self.label_std
                    case "minmax":
                        self.label_min = np.min(train_labels)
                        self.label_max = np.max(train_labels)
                        self.label_range = self.label_max - self.label_min
                        self.label_range = self.label_range if self.label_range != 0 else 1.0
                        for data in [self.training, self.validation, self.testing]:
                            data.labels = (data.labels - self.label_min) / self.label_range