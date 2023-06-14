import uhd
import argparse
from typing import List
import logging


def validate_args(args):
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
    chirp_bw: int = args.chirp_bw
    sampling_rate: int = args.sampling_rate
    center_freqs: List[int] = args.freqs
    min_center_freq = min(center_freqs)
    max_center_freq = max(center_freqs)

    if sampling_rate < MIN_SAMPLING_RATE or sampling_rate > MAX_SAMPLING_RATE:
        valid = False
        msg.append(
            f"Sampling rate {sampling_rate / 1e6} MSps out of range {MIN_SAMPLING_RATE / 1e6}-{MAX_SAMPLING_RATE / 1e6} MSps for device."
        )

    if min_center_freq < MIN_CENTER_FREQ:
        valid = False
        msg.append(
            f"Minimum center frequency {min_center_freq} MHz out of range for device."
        )
    if max_center_freq > MAX_CENTER_FREQ:
        valid = False
        msg.append(
            f"Maximum center frequency {max_center_freq} MHz out of range for device."
        )

    if sampling_rate < (2 + NYQUIST_SAFETY_FACTOR) * chirp_bw:
        valid = False
        msg.append(
            f"Sampling rate {sampling_rate / 1e6} MSps less than Nyquist rate + {NYQUIST_SAFETY_FACTOR*100}% for signal bandwidth {chirp_bw / 1e6} MHz."
        )
    return valid, msg


def main():
    success = False

    args = argparse()
    success, err_msg = validate_args(args)
    if not success:
        logging.error(err_msg)

    return success


if __name__ == "__main__":
    main()
