import pandas as pd
import numpy as np  # Directly import numpy instead of relying on ninc.Util
import logging

def to_data(df: pd.DataFrame, 
            label_column: str = "label", 
            one_hot: bool = False):
    
    features = df.drop(columns=[label_column]).to_numpy()
    if one_hot:
        labels = pd.get_dummies(df[label_column]).to_numpy()
    else:
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
        if self.path.endswith(".csv"):
            self.data = pd.read_csv(self.path)
        elif self.path.endswith(".xlsx"):
            self.data = pd.read_excel(self.path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        logging.info(f"Loading data from {self.path}")
        logging.debug(f"Data preview: {self.data.head()}" if hasattr(self, 'data') else "Data not loaded yet.")

def split(self, shuffle: bool = True, label_column: str = "label", one_hot: bool = False):
    if not hasattr(self, "data"):
        raise RuntimeError("You must call `load()` before `split()`.")

    df = self.data.sample(frac=1, random_state=42).reset_index(drop=True) if shuffle else self.data

    # Check if one-hot encoding is needed
    if one_hot or pd.api.types.is_string_dtype(df[label_column]):
        # Generate one-hot labels and update the DataFrame
        dummies = pd.get_dummies(df[label_column], prefix=label_column)
        df = pd.concat([df.drop(columns=[label_column]), dummies], axis=1)
        label_columns = dummies.columns.tolist()  # List of one-hot column names
    else:
        label_columns = [label_column]  # Single label column

    total_samples = len(df)
    train_end = int(total_samples * self.split_ratio.training)
    val_end = train_end + int(total_samples * self.split_ratio.validation)

    # Use the updated DataFrame and label columns
    self.training = to_data(df.iloc[:train_end], label_columns)
    self.validation = to_data(df.iloc[train_end:val_end], label_columns)
    self.testing = to_data(df.iloc[val_end:], label_columns)

    logging.info(f"Splitting data into training, validation, and testing sets with ratios {self.split_ratio}")
    logging.debug(f"Total samples: {total_samples}, Training samples: {train_end}, Validation samples: {val_end - train_end}, Testing samples: {total_samples - val_end}")
    def split_batches(self, batch_size: int, 
                      shuffle: bool = True) -> list[Data]:
        features, labels = self.training.features, self.training.labels
        if shuffle:
            indices = np.arange(features.shape[0])
            np.random.shuffle(indices)
            features, labels = features[indices], labels[indices]

        self.batches = [
            Data(features=features[i:i+batch_size], labels=labels[i:i+batch_size])
            for i in range(0, features.shape[0], batch_size)
        ]
        return self.batches

    def normalize(self, mode: str = "standardization", normalize_labels: bool = False):
        if not hasattr(self, 'training'):
            raise AttributeError("Data must be split before normalization!")

        self.norm_mode = mode
        train_features = self.training.features

        if mode == "standardization":
            self.feature_mean, self.feature_std = np.mean(train_features, axis=0), np.std(train_features, axis=0)
            self.feature_std[self.feature_std == 0] = 1.0
        elif mode == "minmax":
            self.feature_min, self.feature_max = np.min(train_features, axis=0), np.max(train_features, axis=0)
            self.feature_range = self.feature_max - self.feature_min
            self.feature_range[self.feature_range == 0] = 1.0
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for data in [self.training, self.validation, self.testing]:
            if mode == "standardization":
                data.features = (data.features - self.feature_mean) / self.feature_std
            elif mode == "minmax":
                data.features = (data.features - self.feature_min) / self.feature_range

        if normalize_labels:
            train_labels = self.training.labels
            if mode == "standardization":
                self.label_mean, self.label_std = np.mean(train_labels), np.std(train_labels)
                self.label_std = self.label_std if self.label_std != 0 else 1.0
                for data in [self.training, self.validation, self.testing]:
                    data.labels = (data.labels - self.label_mean) / self.label_std
            elif mode == "minmax":
                self.label_min, self.label_max = np.min(train_labels), np.max(train_labels)
                self.label_range = self.label_max - self.label_min
                self.label_range = self.label_range if self.label_range != 0 else 1.0
                for data in [self.training, self.validation, self.testing]:
                    data.labels = (data.labels - self.label_min) / self.label_range

        logging.info(f"Normalizing data using {mode} mode")
        logging.debug(f"Feature mean: {self.feature_mean if mode == 'standardization' else self.feature_min}, Feature std: {self.feature_std if mode == 'standardization' else self.feature_range}")

        def handle_missing_data(self, 
                        columns: list[str], 
                        strategy: str = "mean", 
                        placeholder: float = 0) -> None:
            """
            Handle missing data in specified columns.
            
            Parameters:
                columns (list[str]): Columns to check for missing values.
                strategy (str): How to handle missing values ("mean", "median", "mode", or "drop").
                placeholder (float): Value to replace (e.g., 0, NaN, or -999).
            """
            if not hasattr(self, "data"):
                raise RuntimeError("You must call `load()` before `handle_missing_data()`.")

            df = self.data

            for col in columns:
                # Replace placeholders (e.g., 0) with NaN for easier handling
                df[col] = df[col].replace(placeholder, np.nan)

                # Apply strategy
                if strategy == "mean":
                    fill_value = df[col].mean()
                elif strategy == "median":
                    fill_value = df[col].median()
                elif strategy == "mode":
                    fill_value = df[col].mode()[0]
                elif strategy == "drop":
                    df.dropna(subset=[col], inplace=True)
                else:
                    raise ValueError(f"Invalid strategy: {strategy}")

                if strategy != "drop":
                    df[col] = df[col].fillna(fill_value)

            self.data = df
            logging.info(f"Handled missing data in columns: {columns}")