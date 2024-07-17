import numpy as np
import matplotlib.pyplot as plt

from rp.tuning.lut import LUT
from rp.tuning.values import calc_sin_steps
from rp.conversions import no_conversion
from rp.core import RedPitayaBoard
from rp.adc.config import get_adc_config_synced
from rp.structs import DacConfig
from rp.dac.config import get_dac_bram_controller_config
from rp.ram.config import get_ram_config, RAM_SIZE
from rp.constants import (
    RP_DAC_PORT_1, 
    RP_DAC_ID, 
    DAC_MODE_STREAM, 
    DAC_CONFIG_ID,
    DAC_BRAM_CONFIG_ID,
    ADC_BLOCK_MODE,
    ADC_CONFIG_ID,
    RAM_INIT_CONFIG_ID,
    )
from rp.adc.receive import AdcDataReceiver
from rp.misc.helpers import create_new_measure_folder

# constants for DAC-config
DAC_STEPS = 125
PORT_ID_CH1 = 0

# ram-size config
RAM_SIZE = RAM_SIZE.KB_512

# adc config
ADC_MODE = ADC_BLOCK_MODE
ADC_DAC_SWEEPS_TO_SAMPLE = 1
START_DELAY = 0

# sampling rate of ADC
FS_MHZ = 125

# amplitude of sine wave 
AMP = 1

PATH_TO_STORE_MEASUREMENT = "./"
N_WORKERS = 8


def init_measurement(pita:RedPitayaBoard, f_input:float, dac_steps:int, amplitude: float):
    
    # generate LUT for selected channel
    dac_lut = LUT(PORT_ID_CH1)
    dac_lut.generateLut(calc_sin_steps, no_conversion, amplitude, dac_steps)
    pita.send_lut_file(dac_lut)
    
    # DAC-config:
    rp_dac_config = DacConfig(
        RP_DAC_ID, DAC_MODE_STREAM, 0, 1)
    
    dwell_time_ms = 1000/(dac_steps * f_input)
    
    # DAC-BRAM-CONTROLLER:
    bram_dac_config = get_dac_bram_controller_config(
        PORT_ID_CH1, 
        rp_dac_config,
        RP_DAC_PORT_1,
        dac_steps,
        dwell_time_ms,
    )
    # send configs for DAC-Device:
    pita.sendConfigParams(rp_dac_config, DAC_CONFIG_ID)
    # send configs for DAC-BRAM-Controller
    pita.sendConfigParams(bram_dac_config, DAC_BRAM_CONFIG_ID)
    print("Configuration set successfully!\n")
    pita.start_dac_sweep(port=PORT_ID_CH1)
    print("DAC-Sweep started")

    return dwell_time_ms


def init_adcram(pita:RedPitayaBoard, dwell_time_ms:float, dac_steps:int, fs:float):   
    
    #get RAM-config:
    ram_config = get_ram_config(RAM_SIZE)

    # get ADC-config:
    adc_config = get_adc_config_synced(
        ADC_DAC_SWEEPS_TO_SAMPLE,
        fs,
        dwell_time_ms,
        dac_steps,
        START_DELAY,
        ADC_MODE,
        ram_config,
        verbose = True,
    )
      # send config for ADC-Module
    pita.sendConfigParams(adc_config["adc"], ADC_CONFIG_ID)

    # ram-config:
    pita.sendConfigParams(ram_config, RAM_INIT_CONFIG_ID)

    return {"adc": adc_config, "ram": ram_config}

def takeMeasurement(
    pita: RedPitayaBoard, adc_data_receiver: AdcDataReceiver, no_tcp_packages: int
):
    # create folder for measurement:
    folder_dir = create_new_measure_folder(PATH_TO_STORE_MEASUREMENT)

    print(f"start sampling {no_tcp_packages} TCP-Packages...")
    # send start sampling command
    pita.start_adc_sampling(no_tcp_packages)

    # start receiving data:
    adc_data_receiver.receive_data(folder_dir, no_tcp_packages)

    print("received all data from RedPitaya")

    return folder_dir

def calculate_average_output(dwell_time_ms: float, periods: int, fs: float, dac_steps: int, adc_samples: list):
    output = []

    # Calculate burst size as an integer
    burst_size = (np.ceil(fs * dwell_time_ms * 1000)
            if fs * dwell_time_ms * 1000 > 1 else 
            1/ np.ceil(1 / (fs * dwell_time_ms * 1000)))

    
    number_of_samples = int(burst_size * periods * dac_steps)

    # Ensure that adc_samples is a list of strings, and split it into a list of first elements
    samples = [sample.split(',')[0] for sample in adc_samples[:number_of_samples]]

    for i in range(dac_steps):
        corresponding_adc_samples = []
        for j in range(periods):
            start_index = int(i * burst_size + j * dac_steps * burst_size )
            end_index = int(start_index + burst_size)

            # Check boundaries
            if start_index < len(samples) and end_index <= len(samples):
                corresponding_adc_samples.extend(samples[start_index:end_index])
            else:
                print(f"Warning: Index out of bounds - start_index: {start_index}, end_index: {end_index}")

        # Convert list to numpy array and compute mean
        corresponding_adc_samples = np.array(corresponding_adc_samples, dtype=float)
        average = np.mean(corresponding_adc_samples) if corresponding_adc_samples.size > 0 else 0
        output.append(average)

    output = np.array(output)
    return output



def calculate_bode_at_f(input_signal, output_signal, f_input, fs):
    """
    Calculate the magnitude and phase at frequency f_input given input and output signals without FFT.

    :param input_signal: Array of input signal values.
    :param output_signal: Array of output signal values.
    :param f_input: Frequency at which to perform the calculation (Hz).
    :param fs: Sampling frequency (Hz).

    :return: Magnitude and Phase (in degrees) at frequency f_input.
    """
    # Length of the signal
    n = DAC_STEPS

    # Generate reference sinusoids at frequency f_input
    t = np.arange(n) / (fs*1e6)
    ref_sin = np.sin(2 * np.pi * f_input * t)
    ref_cos = np.cos(2 * np.pi * f_input * t)

    # Compute dot products for input signal
    input_sin_dot = np.dot(input_signal, ref_sin)
    input_cos_dot = np.dot(input_signal, ref_cos)

    # Compute dot products for output signal
    output_sin_dot = np.dot(output_signal, ref_sin)
    output_cos_dot = np.dot(output_signal, ref_cos)

    # Calculate magnitudes and phases for input and output signals
    input_magnitude = np.sqrt(input_sin_dot**2 + input_cos_dot**2)
    output_magnitude = np.sqrt(output_sin_dot**2 + output_cos_dot**2)

    input_phase = np.arctan2(input_sin_dot, input_cos_dot)
    output_phase = np.arctan2(output_sin_dot, output_cos_dot)

    # Calculate the transfer function H(f_input)
    H_magnitude = output_magnitude / input_magnitude
    H_phase = np.degrees(output_phase - input_phase)

    return H_magnitude, H_phase

if __name__ == "__main__":

    pita = RedPitayaBoard()

    frequencies = np.arange(1e5, 1e6 + 1, 1e5)  # frequencies from 100KHz to 1MHz in 100KHz steps

    magnitudes = []
    phases = []

    for f_input in frequencies:
        dwell_time_ms = init_measurement(pita, f_input, DAC_STEPS, AMP)
        
        adcram_config = init_adcram(pita, dwell_time_ms, DAC_STEPS, FS_MHZ)
        
        adcReceiver = AdcDataReceiver(
            pita,
            adcram_config["ram"],
            n_workers=N_WORKERS,
            verbose=False,
            rawData=False,
        )

        adcReceiver.startWriterThreads()

        folder_dir = takeMeasurement(pita, adcReceiver, adcram_config["adc"]["tcp"])
        
        adcReceiver.finish()

        with open(f'{folder_dir}/measure0001.csv', 'r') as file:
            content  = file.readlines()

        adc_samples = content[1:]

        output_signal = calculate_average_output(dwell_time_ms, 3, FS_MHZ, DAC_STEPS, adc_samples)

        with open('./lut_port0.csv', 'r') as file:
            input_signal = np.array(file.readline().strip().split(','), dtype=float)


        magnitude, phase = calculate_bode_at_f(input_signal, output_signal, f_input, FS_MHZ)

        magnitudes.append(magnitude)
        phases.append(phase)

    # Convert results to numpy arrays
    magnitudes = np.array(magnitudes)
    phases = np.array(phases)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude plot
    ax1.semilogx(frequencies, 20 * np.log10(magnitudes))
    ax1.set_title("Bode Plot")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", ls="--")
    
    # Phase plot
    ax2.semilogx(frequencies, phases)
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", ls="--")
    
    plt.show()
    
            
    pita.close()














