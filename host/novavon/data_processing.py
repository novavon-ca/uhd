# Data processing code for SDR data
# Adapted from Patrick Fiske's University of Vermont MSc Thesis

# Notes:
# - the A-scan matrix contains time-domain signals, each with a different central frequency
# - the matrix is Nfreqs x Nsamples
# - the frequencies range from f1 to fN, with a spacing of dF

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from waveforms import dc_chirp
from scipy.io import loadmat

# Data acqusition parameters
data_from_file: bool = True
sampling_rate: int = 20e6  # [Samples/second]
chirp_bw: int = 10e6  # [Hz]
chirp_duration: float = 50e-6  # [seconds]
num_freqs: int = 3
min_freq: int = 0.9e9
max_freq: int = 1.2e9
center_freqs: np.array = np.linspace(min_freq, max_freq, num_freqs, endpoint=True)
data_filename = "./host/novavon/sample_data/2023-07-05_10-45_20e6_0-9_1-05_1-2.mat"

if data_from_file:
    data = loadmat(data_filename)
    print("Reading ", data["__header__"])
    recv_data_list = data["data"]
    num_samples = recv_data_list.shape[1]
    assert recv_data_list.shape[0] == num_freqs
    time_vector = 1 / sampling_rate * np.linspace(0, num_samples, num=num_samples)
    dt = time_vector[1] - time_vector[0]

    # Plot time-domain data
    legend_text = []
    plt.figure()
    for ii in range(num_freqs):
        plt.plot(time_vector * 1e6, np.real(recv_data_list[ii]))
        legend_text.append(f"signal {ii+1}")
    plt.title("Raw Time-Domain Data")
    plt.xlabel("Time [us]")
    plt.legend(legend_text)
    plt.grid("True")

else:
    # Create some dummy data so we have something to play with
    # TODO: add noise to distort each chirp
    num_tx_samples: int = int(sampling_rate * chirp_duration)
    recv_data_list = []
    time_vector = np.linspace(0, chirp_duration, num=num_tx_samples)
    dt = time_vector[1] - time_vector[0]
    plt.figure()
    for ii, freq in enumerate(center_freqs):
        data_buffer = dc_chirp(0.3, chirp_bw, sampling_rate, chirp_duration, pad=False)
        # data_buffer = np.sin(2 * np.pi * freq * time_vector, dtype=np.complex64)
        recv_data_list.append(data_buffer)
        plt.plot(time_vector, recv_data_list[ii])
    plt.xlabel("Time")

# Frequency-stacking algorithm:
# 1. Take FFT of each baseband sub-pulse: zn[tm] => Zn[fk]
# 2. Apply matched filter to each sub-pulse in frequency domain: Dn[fk] = Zn[fk] * conj(Vn[fk]) where Vn is a reference baseband chirp
# 3. Filter each sub-pulse with window function: Dn[fk] * rect[fk / Bs], where Bs > chirp_bandwidth
# 4. Zero-pad each sub-pulse at both ends s.t. new_num_samples > (num_subpulses * sampling_rate * chirp_duration)
# 5. Shift each sub-pulse to the proper center frequency (circshift)
# 6. Sum the freq-domain sub-pulses
# 7. IFFT
reference_pulse = dc_chirp(1, chirp_bw, sampling_rate, chirp_duration)
reference_pulse = reference_pulse / max(reference_pulse)  # TODO: do we need this?
fd_ref = np.fft.fft(reference_pulse, n=num_samples)
fft_freqs = np.fft.fftfreq(num_samples, d=dt)
len_zero_padding = 2**22 - num_samples  # int(
#     1000 * num_freqs * chirp_duration * sampling_rate
# )  # TODO: decide on this number
summed_sub_pulses_fd = np.empty([num_samples + 2 * len_zero_padding])
# plt.figure()
for ii in range(num_freqs):
    # 1. Take FFT
    # TODO: do we need the normalization and dc-removal?
    td_signal = recv_data_list[ii] / max(recv_data_list[ii])
    td_signal = td_signal - np.mean(td_signal)
    fd_signal = np.fft.fft(td_signal)

    # 2. Apply matched filter
    compressed_fd = fd_signal * np.conj(fd_ref)

    # 3. Filter with rectangular / tukey window
    window = windows.tukey(num_samples, alpha=0.01)
    fd_signal_filt = compressed_fd * window

    # 4. Zero-pad symmetrically about DC
    fd_signal_filt = np.fft.fftshift(fd_signal_filt) # shift DC component to center of spectrum
    fd_signal_filt_padded = np.pad(
        fd_signal_filt, len_zero_padding, "constant", constant_values=(0)
    )
    df = fft_freqs[1] - fft_freqs[0]
    new_max_freq = sampling_rate / 2 + len_zero_padding * df
    # print(np.max(new_max_freq) / 1e6)
    padded_freqs = np.linspace(
        -1 * new_max_freq, new_max_freq, num_samples + 2 * len_zero_padding
    )
    # print(padded_freqs[1:5], padded_freqs[-5:-1])

    # 5. Frequency-shift to center freq
    shift = np.argmin(np.abs(padded_freqs - center_freqs[ii]))
    # print(shift, padded_freqs[shift])
    fd_signal_filt_padded_shifted = np.roll(fd_signal_filt_padded, -shift)

    # 6. Sum sub-pulses
    summed_sub_pulses_fd = summed_sub_pulses_fd + fd_signal_filt_padded_shifted

    # plt.plot(fft_freqs / 1e6, np.abs(fd_signal))
    # plt.plot(fft_freqs / 1e6, np.abs(compressed_fd))
    # plt.plot(np.fft.fftshift(fft_freqs) / 1e6, np.abs(fd_signal_filt))
    # plt.plot(padded_freqs / 1e6, np.abs(fd_signal_filt_padded_shifted))

# 7. IFFT
summed_sub_pulses_td = np.real(np.fft.ifft(summed_sub_pulses_fd))

# plt.legend(
#     [
#         "f1 raw",
#         "f1 compressed",
#         "f1 filt",
#         "f1 shifted"
#         # "f2 raw",
#         # "f2 compressed",
#         # "f2 filt",
#         # "f3 raw",
#         # "f3 compressed",
#         # "f3 filt",
#     ]
# )
# plt.xlabel("Frequencies [MHz]")
# plt.grid()

# plt.figure()
# plt.title("Sum of frequency-domain pulses")
# plt.plot(
#     padded_freqs / 1e6,
#     np.abs(summed_sub_pulses_fd),
# )

plt.figure()
t = np.linspace(0, np.max(time_vector), len(summed_sub_pulses_td))
plt.plot(t, summed_sub_pulses_td)
plt.plot(time_vector, np.sum(recv_data_list, axis=0))
plt.grid()
plt.xlabel("Time")
plt.legend(["Reconstructed", "Sum of original signals"])
plt.show()
