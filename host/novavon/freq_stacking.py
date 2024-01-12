# Frequency stacking algorithm for processing stepped chirp cw radar data

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from waveforms import dc_chirp
from utilities import nextpow2, DataLoader
from typing import List


def time_window(recv_data_list, pulse_duration_samples, plot=False):
    threshold = 0.3
    new_num_samples = int(
        1.0 * pulse_duration_samples
    )  # this might need to change in future
    head = 2000
    num_freqs, num_channels, num_samples = recv_data_list.shape
    data_trimmed = np.empty(
        [num_freqs, num_channels, new_num_samples], dtype=np.complex64
    )
    if plot:
        plt.figure()
    for f_idx in range(num_freqs):
        a = np.real(recv_data_list[f_idx][0])
        b = np.real(recv_data_list[f_idx][1])
        toa0 = np.argwhere(np.abs(a) >= threshold * np.max(abs(a)))[0][0]
        toa1 = np.argwhere(np.abs(b) >= threshold * np.max(abs(b)))[0][0]
        if abs(toa1 - toa0) > head:
            print(f"problem :(   fidx = {f_idx}")
            toa = min([toa0, toa1])
        else:
            toa = int(0.5 * (toa0 + toa1))
        start = min([max([0, toa - head]), num_samples - new_num_samples])
        stop = start + new_num_samples
        data_trimmed[f_idx][0] = recv_data_list[f_idx][0][start:stop]
        data_trimmed[f_idx][1] = recv_data_list[f_idx][1][start:stop]
        if plot:
            plt.cla()
            plt.plot(np.real(data_trimmed[f_idx][0]))
            plt.plot(np.real(data_trimmed[f_idx][1]))
    return data_trimmed


class FrequencyStacking:
    """
    Frequency-stacking algorithm:
    1. Take FFT of each baseband sub-pulse: zn[tm] => Zn[fk]
    2. Apply matched filter to each sub-pulse in frequency domain: Dn[fk] = Zn[fk] * conj(Vn[fk]) where Vn is a phase-coherent reference baseband chirp
    3. Filter each sub-pulse with window function: Dn[fk] * rect[fk / Bs], where Bs > chirp_bandwidth
    4. Zero-pad each sub-pulse at both ends s.t. new_num_samples > (num_subpulses * sampling_rate * chirp_duration)
    5. Shift each sub-pulse to the proper center frequency (circshift)
    6. Sum the freq-domain sub-pulses
    7. IFFT
    """

    def __init__(
        self, recv_data_list: List, acquisition_params: dict, verbose: bool = True
    ) -> None:
        self.verbose = verbose
        self.recv_data_list = recv_data_list
        self.num_freqs, self.num_channels, self.num_samples = recv_data_list.shape
        self.center_freqs = acquisition_params["center_freqs"]
        self.sampling_rate = acquisition_params["sampling_rate"]
        self.chirp_bw = acquisition_params["chirp_bw"]
        self.chirp_duration = acquisition_params["chirp_duration"]

        # Precompute filters and fft freqs
        self.fft_freqs = np.fft.fftfreq(self.num_samples, d=(1 / self.sampling_rate))
        self.fft_freqs_shifted = np.fft.fftshift(self.fft_freqs)
        self.rect_window = np.where(
            np.abs(self.fft_freqs_shifted) <= (self.chirp_bw * 1.0) / 2, 1, 0
        )
        self.ideal_chirp_fd = (
            np.fft.fft(
                dc_chirp(
                    1,
                    self.chirp_bw,
                    self.sampling_rate,
                    self.chirp_duration,
                    pad_length=0,
                ),
                n=self.num_samples,
            )
            / self.num_samples
        )

        self.len_zero_padding = int(
            (max(self.center_freqs) + self.sampling_rate)
            / (self.fft_freqs[1] - self.fft_freqs[0])
        )
        new_max_freq = np.max(self.fft_freqs) + self.len_zero_padding * (
            self.fft_freqs[1] - self.fft_freqs[0]
        )
        self.padded_freqs = np.linspace(
            -1 * new_max_freq,
            new_max_freq,
            num=self.num_samples + 2 * self.len_zero_padding,
            endpoint=False,
        )
        if self.verbose:
            print(f"Max center freq in data: {np.max(self.center_freqs) / 1e6} MHz")
            print(f"Max supported freq after padding: {new_max_freq / 1e6} MHz")

    def compute_sww(self, data_chan: int = 0, reference_chan: int = 1) -> np.ndarray:
        summed_subpulses_fd = np.zeros(
            [self.num_samples + 2 * self.len_zero_padding], dtype=np.complex64
        )

        # Loop over frequencies
        for ii in range(self.num_freqs):
            if self.verbose:
                print(f"Freq {ii+1}/{self.num_freqs}")

            # 1. Compute compressed subpulse spectrum
            compressed_pulse = self.matched_filter(ii, data_chan, reference_chan)

            # 2. Filter baseband pulse with rect window having Bs >= Bi
            fd_signal_filt = np.fft.fftshift(compressed_pulse) * self.rect_window

            # 3. Zero-pad symmetrically about DC
            fd_signal_padded = np.pad(
                fd_signal_filt,
                self.len_zero_padding,
                "constant",
                constant_values=(0),
            )

            # 4. Frequency-shift to correct center freq
            shift = np.argmin(
                np.abs(np.fft.ifftshift(self.padded_freqs) - self.center_freqs[ii])
            )
            fd_signal_padded_shifted = np.roll(fd_signal_padded, shift)

            # 5. Sum sub-pulses
            summed_subpulses_fd = summed_subpulses_fd + fd_signal_padded_shifted

        return summed_subpulses_fd

    def gls_filter(self):
        # TODO
        raise (NotImplementedError)

    def matched_filter(
        self, freq_idx: int, data_chan: int, ref_chan: int
    ) -> np.ndarray:
        if data_chan < self.num_channels:
            # use measured phase-coherent reference
            Zi = np.fft.fft(self.recv_data_list[freq_idx][data_chan]) / self.num_samples
            Vi = np.fft.fft(self.recv_data_list[freq_idx][ref_chan]) / self.num_samples
        else:
            # use ideal chirp
            Zi = self.ideal_chirp_fd
            Vi = self.ideal_chirp_fd
        return Zi * np.conj(Vi)


def main():
    input_dir = "/Users/hannah/Documents/TerraWave/test data/"
    input_filenames = [
        "sfcw_jan10_16Msps_half_chirp_len_A.mat",
        "sfcw_jan10_16Msps_half_chirp_len_B.mat",
        "sfcw_jan10_16Msps_A.mat",
        "sfcw_jan10_16Msps_B.mat",
    ]
    data_ch_idx = 0
    ref_ch_idx = 1
    for ii in range(len(input_filenames)):
        Loader = DataLoader()
        input_filename = input_filenames[ii]
        Loader.load_mat(input_dir + input_filename)
        trimmed_data = time_window(
            Loader.recv_data_list,
            Loader.p["chirp_duration"] * Loader.p["sampling_rate"],
        )
        FreqStacking = FrequencyStacking(trimmed_data, Loader.p, verbose=False)

        sww_spectrum = FreqStacking.compute_sww(data_ch_idx, ref_ch_idx)
        sww_td = np.fft.fftshift(
            len(sww_spectrum) * np.real(np.fft.ifft(np.fft.ifftshift(sww_spectrum)))
        )

        plt.figure(1)
        idx_to_plot = np.nonzero(sww_spectrum)
        plt.plot(
            FreqStacking.padded_freqs[idx_to_plot] / 1e6,
            20
            * np.log10(
                np.abs(sww_spectrum[idx_to_plot])
                / np.max(np.abs(sww_spectrum[idx_to_plot]))
            ),
        )

        plt.figure(2)
        wave_speed = 2e8  # [m/s] -- approx val for coax
        padded_dt = 1 / (2 * np.max(FreqStacking.padded_freqs))
        d = wave_speed * np.arange(
            -len(sww_td) / 2 * padded_dt, len(sww_td) / 2 * padded_dt, padded_dt
        )
        idx_to_plot = np.argwhere(abs(d) < 10)
        plt.plot(
            d[idx_to_plot],
            20 * np.log10(abs(sww_td[idx_to_plot]) / max(abs(sww_td[idx_to_plot]))),
        )

    plt.figure(1)
    plt.title(f"SWW Spectrum")
    plt.ylim([-80, 0])
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Normalized Magnitude [dB]")
    plt.grid()

    plt.figure(2)
    plt.title("Reconstructed A-scan")
    plt.grid()
    plt.xlabel("Range [m]")
    plt.xlim([-5, 5])
    plt.show()


if __name__ == "__main__":
    main()
