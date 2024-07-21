import numpy as np

# Generate a signal for multiple periods
num_periods = 5  # Number of periods you want to consider
single_period_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Single period
signal = np.tile(single_period_signal, num_periods)  # Repeat the period

# Downsampling factor
factor = 2

# Calculate the number of downsampled points per period
n = len(single_period_signal) // factor

# Initialize a list to store downsampled signals
downsampled_signals = []

# Loop over each period and downsample it
for i in range(num_periods):
    start_idx = i * len(single_period_signal)
    end_idx = start_idx + len(single_period_signal)
    period_signal = signal[start_idx:end_idx]
    
    # Downsample the period signal
    downsampled_period_signal = np.mean(period_signal[:n * factor].reshape(-1, factor), axis=1)
    
    # Append the downsampled period signal to the list
    downsampled_signals.append(downsampled_period_signal)

# Convert the list of downsampled signals to a NumPy array
downsampled_signals = np.array(downsampled_signals)

# Calculate the mean across all periods
mean_downsampled_signal = np.mean(downsampled_signals, axis=0)

print("Original signal:", signal)
print("Downsampled signals from each period:", downsampled_signals)
print("Mean downsampled signal across all periods:", mean_downsampled_signal)
