"""
TimeseriesSequencer: A TensorFlow/Keras compatible sequence generator for time series data.

This module provides a generator class that creates sliding window sequences
for training RNN, LSTM, GRU, and other sequence models.
"""

import numpy as np
from tensorflow.keras.utils import Sequence


class TimeseriesSequence(Sequence):
    """
    A Keras Sequence generator for time series data with configurable
    window size, timestep, and prediction horizon.
    
    Parameters
    ----------
    x : np.ndarray
        The input feature time series data. Shape: (n_samples, n_features) or (n_samples,)
    y : np.ndarray
        The target values. Shape: (n_samples,) or (n_samples, n_target_features)
    window_size : int
        The number of time steps to include in each input sequence (lookback period).
    prediction_horizon : int
        The number of time steps ahead to predict (forecast horizon).
    timestep : int, default=1
        The step size between consecutive windows (stride).
        Use timestep=1 for maximum overlap, larger values for less overlap.
    batch_size : int, default=32
        Number of samples per batch.
    shuffle : bool, default=True
        Whether to shuffle the data at the end of each epoch.
    
    Attributes
    ----------
    n_samples : int
        Total number of valid sequences that can be generated.
    
    Example
    -------
    >>> import numpy as np
    >>> from TimeseriesSequence import TimeseriesSequence
    >>> 
    >>> # Create sample data
    >>> x = np.random.randn(1000, 5)  # 1000 timesteps, 5 features
    >>> y = np.random.randn(1000)     # Target variable
    >>> 
    >>> # Create generator with 24-hour lookback, predicting 6 hours ahead
    >>> generator = TimeseriesSequence(
    ...     x=x,
    ...     y=y,
    ...     window_size=24,
    ...     prediction_horizon=6,
    ...     timestep=1,
    ...     batch_size=32
    ... )
    >>> 
    >>> # Use with model.fit()
    >>> # model.fit(generator, epochs=10)
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int = 24,
        prediction_horizon: int = 1,
        timestep: int = 1,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.timestep = timestep
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Handle 1D x data
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        # Ensure y is at least 1D
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        
        # Validate x and y have same length
        if len(self.x) != len(self.y):
            raise ValueError(
                f"x and y must have the same length. Got x: {len(self.x)}, y: {len(self.y)}"
            )
        
        # Calculate valid indices for sequences
        # We need window_size points for input and prediction_horizon points for output
        self.max_index = len(self.x) - self.window_size - self.prediction_horizon + 1
        
        # Generate all valid starting indices with the given timestep
        self.indices = np.arange(0, self.max_index, self.timestep)
        self.n_samples = len(self.indices)
        
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {self.window_size + self.prediction_horizon} "
                f"samples, but got {len(self.x)}."
            )
        
        # Shuffle indices initially if required
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Generate one batch of data.
        
        Parameters
        ----------
        idx : int
            Batch index.
        
        Returns
        -------
        tuple
            (X, y) where X has shape (batch_size, window_size, n_features)
            and y has shape (batch_size, prediction_horizon, n_target_features)
        """
        # Get batch indices
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start:end]
        
        # Initialize arrays
        batch_size = len(batch_indices)
        n_features = self.x.shape[1]
        n_target_features = self.y.shape[1]
        
        X = np.zeros((batch_size, self.window_size, n_features), dtype=np.float32)
        y = np.zeros((batch_size, self.prediction_horizon, n_target_features), dtype=np.float32)
        
        # Fill arrays
        for i, start_idx in enumerate(batch_indices):
            # Input sequence: [start_idx : start_idx + window_size]
            X[i] = self.x[start_idx:start_idx + self.window_size]
            
            # Target sequence: [start_idx + window_size : start_idx + window_size + prediction_horizon]
            target_start = start_idx + self.window_size
            target_end = target_start + self.prediction_horizon
            y[i] = self.y[target_start:target_end]
        
        # Squeeze y if prediction_horizon is 1 and n_target_features is 1
        if self.prediction_horizon == 1 and n_target_features == 1:
            y = y.squeeze()
        elif self.prediction_horizon == 1:
            y = y.squeeze(axis=1)
        elif n_target_features == 1:
            y = y.squeeze(axis=-1)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch if shuffle=True."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_all_data(self) -> tuple:
        """
        Get all sequences at once (useful for validation/testing).
        
        Returns
        -------
        tuple
            (X, y) containing all sequences.
        
        Warning
        -------
        This loads all data into memory. For large datasets, iterate
        through batches instead.
        """
        n_features = self.x.shape[1]
        n_target_features = self.y.shape[1]
        
        X = np.zeros((self.n_samples, self.window_size, n_features))
        y = np.zeros((self.n_samples, self.prediction_horizon, n_target_features))
        
        for i, start_idx in enumerate(self.indices):
            X[i] = self.x[start_idx:start_idx + self.window_size]
            target_start = start_idx + self.window_size
            target_end = target_start + self.prediction_horizon
            y[i] = self.y[target_start:target_end]
        
        # Squeeze if needed
        if self.prediction_horizon == 1 and n_target_features == 1:
            y = y.squeeze()
        elif self.prediction_horizon == 1:
            y = y.squeeze(axis=1)
        elif n_target_features == 1:
            y = y.squeeze(axis=-1)
        
        return X, y
    
    def get_sample_shape(self) -> tuple:
        """
        Get the shape of a single sample.
        
        Returns
        -------
        tuple
            (input_shape, output_shape) for model configuration.
        """
        input_shape = (self.window_size, self.x.shape[1])
        
        if self.prediction_horizon == 1 and self.y.shape[1] == 1:
            output_shape = ()
        elif self.prediction_horizon == 1:
            output_shape = (self.y.shape[1],)
        elif self.y.shape[1] == 1:
            output_shape = (self.prediction_horizon,)
        else:
            output_shape = (self.prediction_horizon, self.y.shape[1])
        
        return input_shape, output_shape
    
    def __repr__(self) -> str:
        return (
            f"TimeseriesSequence(\n"
            f"  n_samples={self.n_samples},\n"
            f"  window_size={self.window_size},\n"
            f"  prediction_horizon={self.prediction_horizon},\n"
            f"  timestep={self.timestep},\n"
            f"  batch_size={self.batch_size},\n"
            f"  n_batches={len(self)},\n"
            f"  x_shape={self.x.shape},\n"
            f"  y_shape={self.y.shape}\n"
            f")"
        )


