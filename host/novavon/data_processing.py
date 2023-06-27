# Data processing code for SDR data
# Adapted from Patrick Fiske's University of Vermont MSc Thesis

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Data acqusition parameters - these are just made up for now
num_samples = 100
num_freqs = 4
min_freq = 100e3
max_freq = 400e3
center_freqs = np.linspace(min_freq, max_freq, num_freqs)

# Create some dummy data so we have something to play with
# Notes:
# - the A-scan matrix contains time-domain signals, each with a different central frequency
# - the matrix is Nfreqs x Nsamples
# - the frequencies range from f1 to fN, with a spacing of dF
# todo: try this with chirps at different center freqs
a_scan_matrix = np.zeros([num_freqs, num_samples])
dt = 1 / (10 * max_freq)
time_vector = dt * np.arange(num_samples)
plt.figure()
for ii, freq in enumerate(center_freqs):
    a_scan_matrix[ii][:60] = np.sin(2 * np.pi * freq * time_vector[:60])
    plt.plot(time_vector, a_scan_matrix[ii])
plt.xlabel("Time")

# Frequency-stacking algorithm:
# 1. Take the FFT of each sub-pulse
# 2. Filter each sub-pulse with window function, where the window is derived from the transmitted signal's center frequency
# 3. Zero-pad each sub-pulse at both ends.
# 4. Shift each sub-pulse to the proper center frequency (circshift)
# 5. Sum the freq-domain sub-pulses
# 6. IFFT
plt.figure()
len_zero_padding = 20  # @todo: how to choose this
summed_sub_pulses_fd = np.zeros([num_samples + len_zero_padding * 2])
window_length = 2
for ii in range(num_freqs):
    fd_signal = np.fft.fft(a_scan_matrix[ii])
    fft_freqs = np.fft.fftfreq(num_samples, d=dt)
    plt.plot(fft_freqs / 1e3, np.abs(fd_signal))
    closest_freq_idx = np.argmin(np.abs(fft_freqs - center_freqs[ii]))
    closest_freq_idx_neg = np.argmin(np.abs(fft_freqs - (-1 * center_freqs[ii])))
    # window = np.zeros(
    #     [
    #         num_samples,
    #     ]
    # )
    # window[closest_freq_idx - window_length : closest_freq_idx + window_length + 1] = 1
    # window[
    #     closest_freq_idx_neg - window_length : closest_freq_idx_neg + window_length + 1
    # ] = 1
    window = window.tukey(num_samples, alpha=0.01)
    # @todo: try using tukey (tapered cosine) instead of square window
    fd_signal_filt = fd_signal * window
    # plt.plot(fft_freqs / 1e3, np.abs(fd_signal_filt), "o--")
    fd_signal_filt = np.fft.fftshift(fd_signal_filt)
    fd_signal_filt_padded = np.concatenate(
        (
            np.zeros([len_zero_padding]),
            fd_signal_filt,
            np.zeros([len_zero_padding]),
        )
    )
    fd_signal_filt_padded = np.fft.ifftshift(fd_signal_filt_padded)
    summed_sub_pulses_fd = summed_sub_pulses_fd + fd_signal_filt_padded
summed_sub_pulses_td = np.real(np.fft.ifft(summed_sub_pulses_fd))

plt.plot(
    # np.fft.fftfreq(num_samples + 2 * len_zero_padding, d=dt) / 1e3,
    fft_freqs / 1e3,
    np.abs(summed_sub_pulses_fd[:100]),
    "k--",
)
plt.legend(["f1", "f2", "f3", "f4", "combined"])
plt.xlabel("Frequencies [kHz]")
plt.grid()

plt.figure()
t = np.linspace(0, np.max(time_vector), len(summed_sub_pulses_td))
plt.plot(t, summed_sub_pulses_td)
plt.plot(
    time_vector,
    a_scan_matrix[0] + a_scan_matrix[1] + a_scan_matrix[2] + a_scan_matrix[3],
)
plt.grid()
plt.xlabel("Time")
plt.legend(["Reconstructed", "Sum of original signals"])
plt.show()
