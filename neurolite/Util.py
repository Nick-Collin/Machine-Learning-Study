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

def norm_input(x: Vector, 
               feature_mean: Vector, 
               feature_std: Vector, 
               norm_mode: str) -> Vector:

    if not (x.shape == feature_mean.shape == feature_std.shape):
        raise ValueError("Shape mismatch in norm_input: x, feature_mean, and feature_std must be the same shape.")

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


def denorm_output(y: Vector, 
                  label_mean: float, 
                  label_std: float, 
                  norm_mode: str) -> Vector:
    
    if label_mean is None or label_std is None:
        logging.error("Labels were not normalized! You shouldn't denormalize outputs.")
        raise ValueError("Labels were not normalized! You shouldn't denormalize outputs.")
    
    if not (y.shape == label_mean.shape == label_std.shape):
        raise ValueError("Shape mismatch in norm_input: x, feature_mean, and feature_std must be the same shape.")

    
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

