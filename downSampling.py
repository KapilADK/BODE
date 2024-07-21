import numpy as np

# Original signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Downsampling factor
factor = 2

# Calculate the number of downsampled points
n = len(signal) // factor

# Downsample the signal
downsampled_signal = np.mean(signal[:n * factor].reshape(-1, factor), axis=1)

print("Original signal:", signal)
print("Downsampled signal:", downsampled_signal)
