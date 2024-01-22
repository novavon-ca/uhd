import logging
from datetime import datetime
from typing import List


def validate_args(min_freq: int, max_freq: int, chirp_bw: int, sampling_rate: int):
    """
    Validates user input arguments for the following:

    - Sampling frequency > 2x chirp bandwidth
    - List of center frequencies in range
    - Tx and Rx settings in range:
    """

    # Limits based on USRP documentation
    MIN_CENTER_FREQ = 70e6
    MAX_CENTER_FREQ = 6e9
    MIN_SAMPLING_RATE = 200e3
    MAX_SAMPLING_RATE = 56e6
    NYQUIST_SAFETY_FACTOR = 0.2  # recommended not to operate within outer 20% of Nyquist zone (host/docs/general.dox)

    valid: bool = True
    msg: List[str] = []

    if sampling_rate < MIN_SAMPLING_RATE or sampling_rate > MAX_SAMPLING_RATE:
        valid = False
        msg.append(
            f"Sampling rate {sampling_rate / 1e6} MSps out of range {MIN_SAMPLING_RATE / 1e6}-{MAX_SAMPLING_RATE / 1e6} MSps for device."
        )

    if min_freq < MIN_CENTER_FREQ:
        valid = False
        msg.append(f"Minimum center frequency {min_freq} MHz out of range for device.")
    if max_freq > MAX_CENTER_FREQ:
        valid = False
        msg.append(f"Maximum center frequency {max_freq} MHz out of range for device.")

    if sampling_rate < (2 + NYQUIST_SAFETY_FACTOR) * chirp_bw:
        valid = False
        msg.append(
            f"Sampling rate {sampling_rate / 1e6} MSps less than Nyquist rate + {NYQUIST_SAFETY_FACTOR*100}% for signal bandwidth {chirp_bw / 1e6} MHz."
        )
    return valid, msg


class LogFormatter(logging.Formatter):
    """Log formatter which prints the timestamp with fractional seconds"""

    @staticmethod
    def pp_now():
        """Returns a formatted string containing the time of day"""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)
        # return "{:%H:%M:%S}".format(now)

    def formatTime(self, record, datefmt=None):
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date


def nextpow2(i):
    """
    Find 2^n that is equal to or greater than i.
    """
    n = 1
    while n < i:
        n *= 2
    return n


class DataLoader:
    def __init__(self, verbose: bool = True, plot_flag: bool = False) -> None:
        self.verbose: bool = verbose
        self.plot_flag: bool = plot_flag
        self.recv_data_list: List = []
        self.p: dict = {
            "num_freqs": 0,
            "num_channels": 0,
            "num_samples": 0,
            "sampling_rate": 0,
            "center_freqs": [],
            "chirp_duration": 0.0,
            "chirp_bw": 0,
        }

    def load_mat(self, filename: str) -> None:
        """
        Read data and acquisition parameters from .mat file
        """
        from scipy.io import loadmat

        data = loadmat(filename)
        if self.verbose:
            print("Reading ", data["__header__"])
        self.recv_data_list = data["data"]
        num_freqs, num_channels, num_samples = self.recv_data_list.shape
        self.p.update(
            {
                "num_freqs": num_freqs,
                "num_channels": num_channels,
                "num_samples": num_samples,
                "sampling_rate": int(data["sampling_rate"]),
                "center_freqs": data["frequencies"].flatten(),
                "chirp_duration": float(data["chirp_duration"]),
                # / int(data["sampling_rate"]) ** 2,  # (fix bug in acquisition code)
                "chirp_bw": int(data["chirp_bw"]),
            }
        )

        if self.plot_flag:
            self.plot_data()

    def plot_data(self):
        import matplotlib.pyplot as plt
        import numpy as np

        n = self.p["num_samples"]
        fs = self.p["sampling_rate"]
        n_freq = self.p["num_freqs"]
        time_vector = 1 / fs * np.linspace(0, n, num=n)

        legend1_text = []
        legend2_text = []
        plt.figure()
        for ii in range(n_freq):
            if ii < n_freq / 2:
                plt.subplot(211)
                legend1_text.append(f"f {ii+1}")
            else:
                plt.subplot(212)
                legend2_text.append(f"f {ii+1}")
            plt.plot(
                time_vector * 1e6,
                np.real(self.recv_data_list[ii][0]),
            )

        plt.title("Raw Time-Domain Data")
        # plt.xlabel("Time [Samples]")
        plt.xlabel("Time [us]")
        plt.legend(legend2_text)
        plt.grid("True")
        plt.subplot(211)
        plt.legend(legend1_text)
        plt.grid("True")
