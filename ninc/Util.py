from Neural_Network import NeuralNetwork
import numpy as np
import logging

Tensor = np.ndarray
Matrix = np.ndarray
Vector = np.ndarray

def is_tensor(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray)

def is_matrix(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 2

def is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1

def norm_input(x: Vector, feature_mean: Vector, feature_std: Vector, norm_mode: str) -> Vector:
    match norm_mode:
        case "standardization":
            if np.any(feature_std == 0):
                logging.warning("Zero found in feature_std during normalization. This will result in inf or nan values.")
            result = (x - feature_mean) / feature_std
            if np.isnan(result).any() or np.isinf(result).any():
                logging.warning("NaN or inf detected in normalized input features.")
            return result
        
        case _:
            raise AttributeError(f"Couldnt resolve norm mode")


def denorm_output(y: Vector, label_mean: float, label_std: float, norm_mode: str) -> Vector:
    if label_mean is None or label_std is None:
        logging.error("Labels were not normalized! You shouldn't denormalize outputs.")
        raise ValueError("Labels were not normalized! You shouldn't denormalize outputs.")
    
    try:
        match norm_mode:
            case "standardization":
                result = y * label_std + label_mean
                if np.isnan(result).any() or np.isinf(result).any():
                    logging.warning("NaN or inf detected in denormalized output.")
                return result
            
            case _:
                logging.error(f"Unknown normalization mode '{norm_mode}' in denorm_output.")
                raise AttributeError(f"Couldn't resolve norm mode '{norm_mode}' in denorm_output.")
            
    except Exception as e:
        logging.error(f"Error in denorm_output: {e}")
        raise

def evaluate(nn: NeuralNetwork):
    '''
    For binary classification models
    '''
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for feature, label in zip(nn.dataset.testing.features, nn.dataset.testing.labels):
        if nn.propagate_forward(feature) >= 0.5: 
            predicted = 1
            if predicted == label:
                true_positives += 1
            else:
                false_positives += 1
        else: 
            predicted = 0
            if predicted == label:
                true_negatives += 1
            else:
                false_negatives += 1
    
    #Precision is the proportion of predicted positives that are actually correct.
    precision = true_positives / (true_positives + false_positives)
    # Recal/Sensivity/True_Positive_Rate  is the proportion of actual positives that were correctly identified.
    recal = true_positives / (true_positives + false_negatives)
    #The F1 score is the harmonic mean of precision and recall. It balances both.
    F1_score = 2 * (precision * recal) / (precision + recal)        

    return (precision, recal, F1_score)