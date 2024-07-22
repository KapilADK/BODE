import numpy as np

def average_adc_samples(dac_steps, num_periods, adc_samples):
    """
    Average ADC samples over a given number of DAC steps and periods.
    
    Parameters:
    - dac_steps: Number of DAC steps
    - num_periods: Number of periods over which the ADC samples are taken
    - adc_samples: Array of ADC samples
    
    Returns:
    - A numpy array containing the mean ADC sample values for each DAC step
    """
    N = len(adc_samples)
    N_p = N // num_periods  # Number of samples per period
    
    mean_adc_samples = np.zeros(dac_steps)
    
    # Calculate number of samples per DAC step in one period
    samples_per_dac_step = N_p // dac_steps
    
    for i in range(dac_steps):
        # Indices for the current DAC step in each period
        start_idx = i * samples_per_dac_step
        end_idx = (i + 1) * samples_per_dac_step
        
        # Collect samples from all periods for the current DAC step
        collected_samples = []
        for period in range(num_periods):
            period_start_idx = period * N_p
            period_end_idx = period_start_idx + N_p
            
            collected_samples.extend(adc_samples[period_start_idx + start_idx:period_start_idx + end_idx])
        
        # Compute the mean of the collected samples
        mean_adc_samples[i] = np.mean(collected_samples)
    
    return mean_adc_samples

# Example usage:
P = 4  # Number of periods
D = 125  # Number of DAC steps

# Simulated ADC samples (for example purposes)
N = 2500  # Total ADC samples for 4 periods
t = np.linspace(0, P * 2 * np.pi, N)
adc_samples = np.sin(t)

mean_adc_samples = average_adc_samples(D, P, adc_samples)

print(mean_adc_samples)
