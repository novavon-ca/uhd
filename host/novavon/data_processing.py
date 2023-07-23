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
data_from_file: bool = False
condense_data: bool = False
exclude_bad_data: bool = False
sampling_rate: int = 25e6  # [Samples/second]
chirp_bw: int = 10e6  # [Hz]
chirp_duration: float = 3e-5  # [seconds]
num_freqs: int = 20
min_freq: int = 1.0e9 + chirp_bw / 2
max_freq: int = 1.2e9 - chirp_bw / 2
center_freqs: np.array = np.linspace(min_freq, max_freq, num_freqs, endpoint=True)
data_filename = (
    "/Users/hannah/Documents/Novavon/Test Data/Reflection1m_25MSps_20steppedChirps.mat"
)
# data_filename = "./host/novavon/sample_data/2023-07-05_10-45_20e6_0-9_1-05_1-2.mat"

if data_from_file:
    data = loadmat(data_filename)
    print("Reading ", data["__header__"])
    recv_data_list = data["data"]

    if condense_data:
        duration_samples = int(chirp_duration * sampling_rate * 2)
        recv_data_list_condensed = np.empty(
            [num_freqs, 2 * duration_samples], dtype=np.complex64
        )
        for ii in range(num_freqs):
            w = recv_data_list[ii]
            peak_sample = np.argmax(np.abs(w))
            start = peak_sample - duration_samples
            stop = peak_sample + duration_samples
            if start < 0:
                start = 0
            if stop >= len(w):
                stop = len(w)
            recv_data_list_condensed[ii] = w[start:stop]
        recv_data_list = recv_data_list_condensed

    num_samples = recv_data_list.shape[1]
    assert recv_data_list.shape[0] == num_freqs
    time_vector = 1 / sampling_rate * np.linspace(0, num_samples, num=num_samples)
    dt = time_vector[1] - time_vector[0]

    # Plot time-domain data
    legend1_text = []
    legend2_text = []
    plt.figure()
    for ii in range(num_freqs):
        if ii < num_freqs / 2:
            plt.subplot(211)
            legend1_text.append(f"signal {ii+1}")
        else:
            plt.subplot(212)
            legend2_text.append(f"signal {ii+1}")
        plt.plot(np.real(recv_data_list[ii]))
        # plt.plot(time_vector * 1e6, np.real(recv_data_list[ii]))

    plt.title("Raw Time-Domain Data")
    plt.xlabel("Time [Samples]")
    # plt.xlabel("Time [us]")
    plt.legend(legend2_text)
    plt.grid("True")
    plt.subplot(211)
    plt.legend(legend1_text)
    plt.grid("True")

else:
    # Generate some dummy data
    zero_pad = True
    if zero_pad:
        num_samples = 2040
    else:
        num_samples: int = int(sampling_rate * chirp_duration)
    recv_data_list = []
    # time_vector = np.linspace(0, chirp_duration, num=num_samples)
    plt.figure()
    for ii, freq in enumerate(center_freqs):
        data_buffer, time_vector = dc_chirp(
            1,
            chirp_bw,
            sampling_rate,
            chirp_duration,
            pad=zero_pad,
            ret_time_samples=True,
        )
        # data_buffer = np.sin(2 * np.pi * freq * time_vector, dtype=np.complex64)
        recv_data_list.append(data_buffer)
        plt.plot(time_vector, recv_data_list[ii])
    dt = time_vector[1] - time_vector[0]
    plt.xlabel("Time")

# Frequency-stacking algorithm:
# 1. Take FFT of each baseband sub-pulse: zn[tm] => Zn[fk]
# 2. Apply matched filter to each sub-pulse in frequency domain: Dn[fk] = Zn[fk] * conj(Vn[fk]) where Vn is a reference baseband chirp
# 3. Filter each sub-pulse with window function: Dn[fk] * rect[fk / Bs], where Bs > chirp_bandwidth
# 4. Zero-pad each sub-pulse at both ends s.t. new_num_samples > (num_subpulses * sampling_rate * chirp_duration)
# 5. Shift each sub-pulse to the proper center frequency (circshift)
# 6. Sum the freq-domain sub-pulses
# 7. IFFT

# 0. Precompute some things
reference_pulse = dc_chirp(1, chirp_bw, sampling_rate, chirp_duration)
fd_ref = np.fft.fft(reference_pulse, n=num_samples) / (len(reference_pulse) / 2.0)
window = windows.tukey(num_samples, alpha=0.01)
fft_freqs = np.fft.fftfreq(num_samples, d=dt)
fft_freqs_shifted = np.fft.fftshift(fft_freqs)
len_zero_padding = 2**18 - num_samples
df = fft_freqs[1] - fft_freqs[0]
new_max_freq = np.max(fft_freqs) + len_zero_padding * df
print("New max freq in MHz after padding:", np.max(new_max_freq) / 1e6)
padded_freqs = np.linspace(
    -1 * new_max_freq,
    new_max_freq,
    num=num_samples + 2 * len_zero_padding,
    endpoint=False,
)
summed_sub_pulses_fd = np.zeros(
    [num_samples + 2 * len_zero_padding], dtype=np.complex64
)
plt.figure()
for ii in range(num_freqs):
    # 1. Take FFT
    # TODO: do we need the normalization and dc-removal?
    td_signal = recv_data_list[ii] / np.max(recv_data_list[ii])
    # td_signal = td_signal - np.mean(td_signal)
    fd_signal = np.fft.fft(td_signal) / (len(td_signal) / 2.0)
    print(ii + 1, np.mean(np.abs(fd_signal)))

    if exclude_bad_data:
        if np.mean(np.abs(fd_signal)) > 0.012 or np.mean(np.abs(fd_signal)) < 0.009:
            continue

    # 2. Apply matched filter
    compressed_fd = fd_signal * np.conj(fd_ref)

    # 3. Filter with rectangular / tukey window
    fd_signal_filt = np.fft.fftshift(compressed_fd) * window

    # 4. Zero-pad symmetrically about DC
    fd_signal_padded = np.pad(
        fd_signal_filt,
        len_zero_padding,
        "constant",
        constant_values=(0),
    )

    # 5. Frequency-shift to correct center freq
    shift = np.argmin(np.abs(np.fft.ifftshift(padded_freqs) - center_freqs[ii]))
    fd_signal_padded_shifted = np.roll(fd_signal_padded, shift)

    # 6. Sum sub-pulses
    summed_sub_pulses_fd = summed_sub_pulses_fd + fd_signal_padded_shifted

    # plt.plot(fft_freqs / 1e6, np.abs(fd_signal))
    # plt.plot(fft_freqs_shifted / 1e6, np.abs(np.fft.fftshift(compressed_fd)), "--")
    # plt.plot(fft_freqs_shifted / 1e6, window, "o")
    # plt.plot(fft_freqs_shifted / 1e6, np.abs(fd_signal_filt), "--")
    # plt.plot(padded_freqs / 1e6, np.abs(fd_signal_padded))
    plt.plot(padded_freqs / 1e6, 20 * np.log10(np.abs(fd_signal_padded_shifted)))

# 7. IFFT
# window = windows.tukey(len(summed_sub_pulses_fd), alpha=0.01)
summed_sub_pulses_td = np.real(np.fft.ifft(np.fft.ifftshift(summed_sub_pulses_fd)))

plt.title("Stacked SWW")
plt.xlabel("Frequencies [MHz]")
plt.ylabel("Magnitude [dB]")
# plt.xlim((min_freq - 10 * chirp_bw) / 1e6, (max_freq + 10 * chirp_bw) / 1e6)
plt.grid()

plt.figure()
plt.title("Synthetic wideband waveform")
plt.plot(
    padded_freqs,
    20 * np.log10(np.abs(summed_sub_pulses_fd)),
)
# plt.xlim(min_freq - 10 * chirp_bw, max_freq + 10 * chirp_bw)
plt.xlabel("Freq [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid()


plt.figure()
plt.title("Reconstructed A-scan")
wave_speed = 3e8  # [m/s]
d = wave_speed * np.arange(0, len(summed_sub_pulses_td) * dt, step=dt)
plt.plot(d, summed_sub_pulses_td)
plt.grid()
plt.xlabel("Range [m]")
plt.show()
