import logging
from datetime import datetime
from typing import List, Tuple


def validate_args(
    min_freq: int, max_freq: int, chirp_bw: int, sampling_rate: int
) -> Tuple(bool, str):
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
