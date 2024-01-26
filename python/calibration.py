from utilities import DataLoader, trim_time_domain_data
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List


class GainCompensation:
    def __init__(self, data_file: str) -> None:
        Loader = DataLoader()
        Loader.load_mat(data_file)
        self.trimmed_data = trim_time_domain_data(
            Loader.recv_data_list,
            Loader.p["chirp_duration"] * Loader.p["sampling_rate"],
        )
        self.center_freqs = Loader.p["center_freqs"]
        self.subpulse_width = Loader.p["chirp_bw"]
        self.fs = Loader.p["sampling_rate"]
        _, num_channels, self.num_samples = self.trimmed_data.shape
        self.fft_freqs = np.fft.fftshift(np.fft.fftfreq(self.num_samples, 1 / self.fs))
        self.max_vals = np.zeros(
            [
                num_channels,
            ]
        )

    def calc_gain_factors(self, rx_channel_idx) -> np.ndarray:
        max_vals = []
        for ii in range(len(self.center_freqs)):
            y = np.abs(
                np.fft.fft(self.trimmed_data[ii][rx_channel_idx]) / self.num_samples
            )
            y[:10] = 0.00001
            y[-10:] = 0.00001
            max_vals.append(max(y))

        self.max_vals[rx_channel_idx] = max(max_vals)
        return 20 * np.log10(np.array(max_vals) / max(max_vals))

    def plot_gain_factors(self, rx_channel_idx, cal_factors_db) -> None:
        plt.figure()
        for ii, rf_freq in enumerate(self.center_freqs):
            y = np.abs(
                np.fft.fft(self.trimmed_data[ii][rx_channel_idx]) / self.num_samples
            )
            plt.plot(
                self.fft_freqs + rf_freq,
                20 * np.log10(np.fft.fftshift(y) / self.max_vals[rx_channel_idx]),
            )
        plt.plot(self.center_freqs, cal_factors_db, "k")
        plt.xlabel("Freq [Hz]")
        plt.ylabel("Mag [dB]")
        plt.ylim([-25, 0])
        plt.grid(True)

    def write_gain_table(self, filename: str, cal_factors: List) -> None:
        try:
            with open(filename, "r") as f:
                gain_table = json.load(f)
        except:
            gain_table = {}
        gain_table.update(
            {
                f"{str(self.subpulse_width)}": {
                    "freqs": self.center_freqs.tolist(),
                    "factors": cal_factors,
                }
            }
        )
        with open(filename, "w") as f:
            json.dump(gain_table, f)


if __name__ == "__main__":
    input_dir = "/Users/hannah/Documents/TerraWave/test data/"
    input_filename_A = "sfcw_jan22_15mhzchirp_A.mat"
    input_filename_B = "sfcw_jan22_15mhzchirp_B.mat"
    cal_filename = "/Users/hannah/Documents/TerraWave/test data/gain_tables.json"
    plot_flag = True

    CalA = GainCompensation(input_dir + input_filename_A)
    CalB = GainCompensation(input_dir + input_filename_B)
    ch0 = np.round(0.5 * (CalA.calc_gain_factors(0) + CalB.calc_gain_factors(0)))
    ch1 = np.round(0.5 * (CalA.calc_gain_factors(1) + CalB.calc_gain_factors(1)))
    if plot_flag:
        CalA.plot_gain_factors(0, ch0)
        CalA.plot_gain_factors(1, ch1)
        CalB.plot_gain_factors(0, ch0)
        CalB.plot_gain_factors(1, ch1)
        plt.show()
    if min(ch1) < -20:
        problems = np.argwhere(ch1 <= -20)[0]
        for p in problems:
            ch1[p] = np.mean(ch1[np.array([p - 1, p + 1]).flatten()])
    if min(ch0) < -20:
        problems = np.argwhere(ch0 <= -20)[0]
        for p in problems:
            ch0[p] = np.mean(ch0[np.array([p - 1, p + 1]).flatten()])
    CalA.write_gain_table(cal_filename, [abs(ch0).tolist(), abs(ch1).tolist()])
