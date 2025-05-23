
# Study and implement regularization logics:
    [ ]L1
    [ ]L2
    [ ]ropout	
    [x]Early stopping 
    [ ]BatchNorm	
    [ ]Data augmentation	
    [ ]Label smoothing	
    [ ]Ensemble

# Fix bugs
    [ ]Activation.Softmax()
    [ ]Activation.Softplus()
    [ ]Activation batch shape check
    [ ]Neural_Network.propagate_backward()
    [ ]DataHandeling sanity checks
    

# Add weight initialization module

# Add warning if a lot missing/corrupted data was found

# Add unit tests
    [ ]Activation
    [ ]Cost
    [ ]DataHandling
    [ ]Evaluators
    [ ]Neural_Network
    [ ]Optimizer
    [ ]Trainer
    [ ]Util

# Code examples
    [ ]Diabetes
    [ ]Iris
    [ ]Penguins
    [ ]Wine

# One-hot encoding logic repeated in DataHandeling

The one-hot logic appears in both split() and to_data(), which could lead to inconsistent handling.
Improvement: Consider separating the one-hot conversion into its own method to avoid redundancy and reduce coupling.

# Support saving/loading Dataset splits

You may eventually want to add methods like save_splits() and load_splits() to persist preprocessed data and avoid redoing the split/loading every run.

# Allow setting a random_state for reproducibility in DataHandeling.split_batches()

Currently, batches can be shuffled, but reproducibility depends on global RNG state.
Fix:

```python
def split_batches(self, batch_size: int, shuffle: bool = True, seed: int = None):
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        ...
```