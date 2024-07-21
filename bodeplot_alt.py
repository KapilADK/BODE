import numpy as np
import matplotlib.pyplot as plt

def compute_bode_plot(input_signal, output_signal, fs):
    """
    Compute the Bode plot from input and output signals.

    :param input_signal: Array of input signal values.
    :param output_signal: Array of output signal values.
    :param fs: Sampling frequency (Hz).

    :return: Frequencies, magnitudes (dB), and phases (degrees).
    """
    # Compute the number of samples
    n = len(input_signal)
    
    # Compute the Fourier Transforms
    freq = np.fft.fftfreq(n, d=1/fs)
    input_fft = np.fft.fft(input_signal)
    output_fft = np.fft.fft(output_signal)

    # Compute the magnitude and phase of the transfer function
    H_fft = output_fft / input_fft
    magnitude = np.abs(H_fft)
    phase = np.angle(H_fft, deg=True)

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(magnitude)

    # Remove negative frequencies
    positive_freqs = freq > 0
    freq = freq[positive_freqs]
    magnitude_db = magnitude_db[positive_freqs]
    phase = phase[positive_freqs]

    return freq, magnitude_db, phase

def plot_bode(freq, magnitude_db, phase):
    """
    Plot the Bode diagram.

    :param freq: Array of frequencies.
    :param magnitude_db: Magnitude in dB.
    :param phase: Phase in degrees.
    """
    plt.figure()

    # Plot magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, magnitude_db)
    plt.title('Bode Plot')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both')

    # Plot phase
    plt.subplot(2, 1, 2)
    plt.semilogx(freq, phase)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, which='both')

    plt.show()

# Example usage
fs = 1e6  # 1 MHz sampling frequency
t = np.arange(0, 1, 1/fs)  # 1 second duration
input_signal = np.sin(2 * np.pi * 1e3 * t)  # 1 kHz sine wave as input signal
output_signal = 0.5 * np.sin(2 * np.pi * 1e3 * t + np.pi / 4)  # 1 kHz sine wave with phase shift

# Compute the Bode plot
freq, magnitude_db, phase = compute_bode_plot(input_signal, output_signal, fs)

# Plot the Bode diagram
plot_bode(freq, magnitude_db, phase)
