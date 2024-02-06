# Frequency stacking algorithm for processing stepped chirp cw radar data

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, hilbert
from waveforms import dc_chirp, chirp
from utilities import trim_time_domain_data, DataLoader
from typing import List
from enum import Enum, auto


class WindowType(Enum):
    RECT = auto()
    HAMMING = auto()
    TUKEY = auto()


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
            np.abs(self.fft_freqs_shifted) <= (self.chirp_bw * 1.02) / 2, 1, 0
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
            compressed_pulse = self.matched_filter(
                ii, data_chan, reference_chan, meas_phs_only=True
            )

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

    def gls_filter(
        self, window_type: WindowType = WindowType.RECT, synthetic: bool = True
    ):
        if synthetic:
            chan_idx_synth = np.inf
            W = self.compute_sww(chan_idx_synth, chan_idx_synth)
        else:
            # TODO: build reference SWW from calibration rx channel
            raise NotImplementedError
        min_freq = self.center_freqs[0] - self.chirp_bw / 2
        max_freq = self.center_freqs[-1] + self.chirp_bw / 2
        fs_wb = len(W) * (self.padded_freqs[1] - self.padded_freqs[0])
        M = np.fft.fftshift(
            np.fft.fft(
                chirp(
                    fs_wb,
                    self.chirp_duration,
                    min_freq,
                    max_freq,
                ),
                n=len(W),
            )
        ) / len(W)
        if window_type == WindowType.RECT:
            window = (
                np.where(np.abs(self.padded_freqs) <= max_freq, 1, 0)
                + np.where(np.abs(self.padded_freqs) >= min_freq, 1, 0)
                - 1
            )
        elif window_type == WindowType.HAMMING:
            window = windows.hamming(len(W))
        else:
            raise NotImplementedError
        nonzeros = np.nonzero(W)
        H = np.zeros_like(W)
        H[nonzeros] = M[nonzeros] / W[nonzeros] * window[nonzeros]
        return H / np.max(np.abs(H))

    def matched_filter(
        self,
        freq_idx: int,
        data_chan: int,
        ref_chan: int,
        meas_phs_only: bool = False,
    ) -> np.ndarray:
        # if channel indices are in range of real connected channels, use experimental data; otherwise use ideal chirp
        Zi = (
            np.fft.fft(self.recv_data_list[freq_idx][data_chan]) / self.num_samples
            if data_chan < self.num_channels
            else self.ideal_chirp_fd
        )
        Vi = (
            np.fft.fft(self.recv_data_list[freq_idx][ref_chan]) / self.num_samples
            if ref_chan < self.num_channels
            else self.ideal_chirp_fd
        )
        Di = Zi * np.conj(Vi)
        if meas_phs_only:
            return np.abs(Zi * self.ideal_chirp_fd) * np.exp(1j * np.angle(Di))
        else:
            return Di


def main():
    input_dir = "/Users/hannah/Documents/TerraWave/test data/Feb 5 Range Testing/"
    input_filenames = [
        # "emptyRoom_greenAnt_2ftback.mat",
        # "inside_5ft_greenAnt_2ftback.mat",
        "emptyRoom_greenAnt.mat",
        "inside_3ft_greenAnt.mat",
        "inside_4ft_greenAnt.mat",
        "inside_5ft_greenAnt.mat",
        "inside_8ft_greenAnt.mat",
        "inside_10ft_greenAnt.mat",
    ]
    data_ch_idx = 1
    ref_ch_idx = 0
    time_domain_analysis = True
    gls_flag = False
    subtract_cal_scan = True
    subtract_cable_len_ft = (93.5 + 110 - 12) / 12

    speed_in_cable = 2e8
    speed_in_air = 3e8  # [m/s]
    ft_per_meter = 3.28
    num_rows = 2
    num_cols = int(np.ceil(len(input_filenames) / 2))

    for ii in range(len(input_filenames)):
        Loader = DataLoader()
        input_filename = input_filenames[ii]
        Loader.load_mat(input_dir + input_filename)
        trimmed_data = trim_time_domain_data(
            Loader.recv_data_list,
            Loader.p["chirp_duration"] * Loader.p["sampling_rate"],
        )

        FreqStacking = FrequencyStacking(trimmed_data, Loader.p, verbose=True)
        sww_spectrum = FreqStacking.compute_sww(data_ch_idx, ref_ch_idx)
        if gls_flag:
            gls_filter = FreqStacking.gls_filter()
            sww_spectrum = gls_filter * sww_spectrum

        plt.figure(1)
        plt.subplot(num_rows, num_cols, ii + 1)
        plt.title(input_filename)
        idx_to_plot = np.nonzero(sww_spectrum)
        plt.plot(
            FreqStacking.padded_freqs[idx_to_plot] / 1e6,
            20
            * np.log10(
                np.abs(sww_spectrum[idx_to_plot])
                / np.max(np.abs(sww_spectrum[idx_to_plot]))
            ),
        )
        plt.grid()

        if time_domain_analysis:
            sww_td = np.fft.fftshift(
                len(sww_spectrum) * np.real(np.fft.ifft(np.fft.ifftshift(sww_spectrum)))
            )
            if subtract_cal_scan:
                if ii == 0:
                    cal_td = sww_td
                else:
                    sww_td = sww_td - cal_td

            plt.figure(2)
            plt.subplot(num_rows, num_cols, ii + 1)
            plt.title(input_filename)

            padded_dt = 1 / (2 * np.max(FreqStacking.padded_freqs))
            time_vec = np.arange(
                -len(sww_td) / 2 * padded_dt, len(sww_td) / 2 * padded_dt, padded_dt
            )

            if subtract_cable_len_ft:
                extra_travel_time = subtract_cable_len_ft / (
                    speed_in_cable * ft_per_meter
                )
                time_vec = time_vec - extra_travel_time

            d = speed_in_air * time_vec / 2  # account for 2-way travel time
            idx_to_plot = np.argwhere(abs(d) < 10)
            sww_td_env = abs(hilbert(sww_td))
            d = d * ft_per_meter  # convert m to ft

            plt.plot(
                d[idx_to_plot],
                20 * np.log10(sww_td_env[idx_to_plot] / max(sww_td_env[idx_to_plot])),
            )
            plt.grid()
            plt.xlim([0, 30])
            plt.ylim([-40, 2])

            plt.figure(3)
            plt.plot(
                d[idx_to_plot],
                20 * np.log10(sww_td_env[idx_to_plot] / max(sww_td_env[idx_to_plot])),
            )

    plt.figure(1)
    plt.suptitle(f"CH1 SWW Spectrum")
    plt.ylim([-80, 0])
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Normalized Magnitude [dB]")
    # plt.grid()

    if time_domain_analysis:
        plt.figure(2)
        plt.suptitle("Reconstructed A-scans")
        # plt.grid()
        plt.xlabel("Range [ft]")

        plt.figure(3)
        plt.suptitle("Reconstructed A-scans")
        plt.grid(True)
        plt.xlabel("Range [ft]")
        plt.legend(["empty", "3ft", "4ft", "5ft", "8ft", "10ft"], loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
