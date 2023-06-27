import uhd
import argparse
import logging
import numpy as np
import sys
import threading
from typing import Any, List

from scipy.io import savemat

from utilities import LogFormatter, validate_args
from waveforms import dc_chirp
from usrp_settings import usrp_setup, tx_setup, rx_setup


def tune_center_freq(usrp, center_freq):
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq, 0))
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq, 0))

    while not (
        usrp.get_rx_sensor("lo_locked", 0).to_bool()
        and usrp.get_tx_sensor("lo_locked", 0).to_bool()
    ):
        pass

    return


def tx_worker(streamer, buf, metadata, waveform):
    total_num_samps = np.size(waveform, 0)
    num_tx_samps = 0
    num_acc_samps = 0
    buf = (
        waveform  # TODO: fix this - might want to work for waveform of arbitrary length
    )

    while num_acc_samps < total_num_samps:
        num_tx_samps += streamer.send(buf, metadata)
        num_acc_samps += min(
            total_num_samps - num_acc_samps, streamer.get_max_num_samps()
        )
        metadata.has_time_spec = False
        metadata.start_of_burst = False

    # Send a mini EOB packet
    metadata.end_of_burst = True
    streamer.send(np.zeros((1, 0), dtype=np.complex64), metadata)


def rx_worker(usrp, streamer, buf, metadata, recv_data):
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(
        usrp.get_time_now().get_real_secs() + RX_DELAY
    )
    streamer.issue_stream_cmd(stream_cmd)

    num_rx_samps: int = 0
    total_samples: int = len(recv_data)
    num_samples_per_packet = streamer.get_max_num_samps()

    # Receive until set number of samples are captured
    for ii in range(total_samples // num_samples_per_packet):
        try:
            rx = streamer.recv(buf, metadata)
            recv_data[
                ii * num_samples_per_packet : (ii + 1) * num_samples_per_packet
            ] = buf[0]
            num_rx_samps += int(rx)

        except RuntimeError as ex:
            logger.error("Runtime error in receive: %s", ex)
            return

    # Issue a stop command
    streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))


def main():
    # Settings from user - these will come from the command line or a JSON file
    min_freq: int = 1e9  # [Hz]
    max_freq: int = 3e9  # [Hz]
    num_freqs: int = 4
    chirp_bw: int = 5e6  # [Hz]
    chirp_duration: int = 1e-6  # [seconds] TODO: calculate a reasonable value for this
    output_filename: str = ""  # set to empty string to not save data to file
    verbose: bool = False

    # Settings the user will not have access to
    sampling_rate: int = 20e6  # samples per second
    chirp_ampl: float = 0.3  # float between 0 and 1
    tx_gain: int = 60  # [dB]
    rx_gain: int = 50  # [dB]
    rx_samples: int = 50000
    rx_auto_gain: bool = False
    plot_data: bool = True

    # Validate input args
    success, err_msg = validate_args(min_freq, max_freq, chirp_bw, sampling_rate)
    if not success:
        logging.error(err_msg)

    # Set up USRP device and generate tx waveform
    center_freqs = np.linspace(min_freq, max_freq, num_freqs, endpoint=True)

    setup_args = {
        "tx_rate": sampling_rate,
        "tx_gain": tx_gain,
        "center_freq": min_freq,
        "rx_rate": sampling_rate,
        "rx_gain": rx_gain,
        "rx_auto_gain": rx_auto_gain,
    }
    usrp = usrp_setup(setup_args, logger, verbose)
    waveform = dc_chirp(chirp_ampl, chirp_bw, chirp_duration, sampling_rate)

    # Set up transmit and receive streamers
    tx_streamer, tx_buf, tx_metadata = tx_setup(usrp)
    rx_streamer, recv_buf, rx_metadata = rx_setup(usrp)
    recv_data = np.zeros(num_freqs, rx_samples, dtype=np.complex64)

    # Create tx and rx threads
    threads: List[threading.Thread] = []
    rx_thread = threading.Thread(
        target=rx_worker, args=(usrp, rx_streamer, recv_buf, rx_metadata, recv_data)
    )
    threads.append(rx_thread)
    rx_thread.setName("rx_stream")

    tx_thread = threading.Thread(
        target=tx_worker, args=(tx_streamer, tx_buf, tx_metadata, waveform)
    )
    threads.append(tx_thread)
    tx_thread.setName("tx_stream")

    # Loop through frequencies
    recv_data_list = []
    for freq_idx, frequency in enumerate(center_freqs):
        logger.info(f"Acquiring data at center freq {freq_idx+1}/{num_freqs}...")

        tune_center_freq(usrp, frequency)

        for thr in threads:
            thr.start()
            thr.join()

        recv_data_list.append(recv_data)

    logger.info("Acquisition complete!")

    if len(output_filename):
        logger.info("Saving to file...")
        savemat(output_filename, {"data": recv_data_list})
        logger.info(f"Data written to {output_filename}")

    if plot_data:
        if verbose:
            logger.info("Plotting received data...")

        legend_text = ["Transmitted"]
        import matplotlib.pyplot as plt

        plt.figure()
        for ii in range(num_freqs):
            plt.plot(np.real(recv_data_list[ii]))
            legend_text.append(f"Rx {ii+1}")
        plt.legend(legend_text)
        plt.show()


if __name__ == "__main__":
    RX_DELAY = 0.01  # offset delay between transmitting and receiving

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)
    formatter = LogFormatter(
        fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s"
    )
    console.setFormatter(formatter)

    sys.exit(not main())
