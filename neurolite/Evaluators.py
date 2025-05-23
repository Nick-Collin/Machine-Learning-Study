from abc import ABC, abstractmethod
from neurolite.Neural_Network import NeuralNetwork
from neurolite.DataHandling import Data
import numpy as np
import logging

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model, dataset):
        raise NotImplementedError("Must be implemented by subclass.")

class BinaryClassificationEvaluator(Evaluator):
    def evaluate(model: NeuralNetwork, dataset: Data, threshold: float = 0.5):
        assert hasattr(dataset, "features") and hasattr(dataset, "labels"), "Dataset must have been initialized before evaluation"
        tp = fp = tn = fn = 0

        if len(dataset.features) != len(dataset.labels):
            raise ValueError("Mismatch between number of features and labels.")

        for x, y in zip(dataset.features, dataset.labels):
            pred_val = model.propagate_forward(x)

            pred_class = 1 if pred_val >= threshold else 0

            if pred_class == 1 and y == 1:
                tp += 1
            elif pred_class == 1 and y == 0:
                fp += 1
            elif pred_class == 0 and y == 0:
                tn += 1
            elif pred_class == 0 and y == 1:
                fn += 1

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1}

class RegressionEvaluator(Evaluator):
    def evaluate(model: NeuralNetwork, dataset: Data):
        errors = []
        squared_errors = []
        total = len(dataset.features)
        y_true = []
        y_pred = []

        if len(dataset.features) != len(dataset.labels):
            raise ValueError("Mismatch between number of features and labels.")

        for x, y in zip(dataset.features, dataset.labels):
            pred_class = model.propagate_forward(x)
            y_true.append(y)
            y_pred.append(pred_class)
            error = abs(pred_class - y)
            errors.append(error)
            squared_errors.append((pred_class - y) ** 2)

        mae = sum(errors) / total
        mse = sum(squared_errors) / total
        rmse = (mse) ** 0.5
        mean_y = sum(y_true) / total
        denom = sum((yi - mean_y) ** 2 for yi in y_true)

        if denom == 0: 
            r2 = 1 
            logging.WARNING("Division by zero in r2 evaluation")
        else: 
            r2 = 1 - (sum(squared_errors) / denom)


        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    
class MulticlassEvaluator(Evaluator):
    def evaluate(model, dataset):
        from collections import defaultdict

        num_classes = len(dataset.labels[0])
        confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        
        if len(dataset.features) != len(dataset.labels):
            raise ValueError("Mismatch between number of features and labels.")

        for x, y_true_oh in zip(dataset.features, dataset.labels):
            if sum(y_true_oh) != 1:
                raise ValueError("Multiclass labels must be one-hot encoded.")

            y_true = int(np.argmax(y_true_oh)) # get class index from one-hot
            y_pred = model.propagate_forward(x)
            y_pred_class = int(np.argmax(y_pred))
            
            confusion[y_true][y_pred_class] += 1

        # Per-class metrics
        precision_list = []
        recall_list = []
        f1_list = []

        for cls in range(num_classes):
            tp = confusion[cls][cls]
            fp = sum(confusion[r][cls] for r in range(num_classes) if r != cls)
            fn = sum(confusion[cls][c] for c in range(num_classes) if c != cls)

            if tp + fp == 0: logging.WARNING("Division by zero in evaluation, ading epsilon (1e-8)")

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Macro average
        macro_precision = sum(precision_list) / num_classes
        macro_recall = sum(recall_list) / num_classes
        macro_f1 = sum(f1_list) / num_classes

        return {
            "confusion_matrix": confusion,
            "precision_per_class": precision_list,
            "recall_per_class": recall_list,
            "f1_per_class": f1_list,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1
        }


