import uhd
import argparse
import logging
import numpy as np
import sys
import threading
from typing import Any, List
import matplotlib.pyplot as plt
from scipy.io import savemat

from utilities import LogFormatter, validate_args
from waveforms import dc_chirp
from usrp_settings import usrp_setup, setup_streamers


def tune_center_freq(usrp, center_freq):
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq, 0))
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq, 0))

    while not (
        usrp.get_rx_sensor("lo_locked", 0).to_bool()
        and usrp.get_tx_sensor("lo_locked", 0).to_bool()
    ):
        pass

    return


def tx_worker(streamer, metadata, tx_data, verbose=False):
    total_num_samps = tx_data.size
    num_tx_samps = 0
    num_acc_samps = 0
    while num_acc_samps < total_num_samps:
        num_tx_samps += streamer.send(tx_data, metadata)
        num_acc_samps += min(
            total_num_samps - num_acc_samps, streamer.get_max_num_samps()
        )
        metadata.has_time_spec = False
        metadata.start_of_burst = False

    # Send a mini EOB packet
    metadata.end_of_burst = True
    streamer.send(np.zeros((1, 0), dtype=np.complex64), metadata)
    if verbose:
        logger.info(f"# tx samples: {num_acc_samps}")


def rx_worker(usrp, streamer, metadata, rx_data, verbose=False):
    num_rx_samps: int = 0
    total_samples: int = len(rx_data)
    num_samples_per_packet = streamer.get_max_num_samps()
    recv_buffer: np.ndarray = np.empty((1, num_samples_per_packet), dtype=np.complex64)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    stream_cmd.time_spec = uhd.types.TimeSpec(
        usrp.get_time_now().get_real_secs() + RX_DELAY
    )
    streamer.issue_stream_cmd(stream_cmd)

    # Receive until set number of samples are captured
    for ii in range(total_samples // num_samples_per_packet):
        try:
            rx = streamer.recv(recv_buffer, metadata)
            rx_data[
                ii * num_samples_per_packet : (ii + 1) * num_samples_per_packet
            ] = recv_buffer[0]
            num_rx_samps += int(rx)

        except RuntimeError as ex:
            logger.error("Runtime error in receive: %s", ex)
            return

    # Issue a stop command
    streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
    if verbose:
        logger.info(f"# rx samples: {num_rx_samps}")


def main():
    # Settings from user - these will come from the command line or a JSON file
    chirp_bw: int = 8e6  # [Hz]
    chirp_duration: int = 1.025e-5  # [seconds]
    min_freq: int = 1.4e9 + chirp_bw/2  # [Hz]
    max_freq: int = 1.56e9 - chirp_bw /2 # [Hz]
    num_freqs: int = 3
    output_filename: str = "Demo"  # "2023-07-05_10-45_20e6_0-9_1-05_0-2"  # set to empty string to not save data to file
    verbose: bool = False

    # Settings the user will not have access to
    sampling_rate: int = 25e6  # samples per second
    chirp_ampl: float = 0.3  # float between 0 and 1
    tx_gain: int = 40  # [dB]
    rx_gain: int = 40  # [dB]
    rx_samples: int = 80000
    rx_auto_gain: bool = False
    plot_data: bool = True
    num_averages: int = 3

    # Validate input args
    center_freqs = np.linspace(min_freq, max_freq, num_freqs, endpoint=True)
    success, err_msg = validate_args(min_freq, max_freq, chirp_bw, sampling_rate)
    if not success:
        logger.error(err_msg)
        return

    # Set up USRP device
    setup_args = {
        "tx_rate": sampling_rate,
        "tx_gain": tx_gain,
        "center_freq": min_freq,
        "rx_rate": sampling_rate,
        "rx_gain": rx_gain,
        "rx_auto_gain": rx_auto_gain,
    }
    usrp = usrp_setup(setup_args, logger, verbose)

    # Set up transmit and receive streamers
    tx_streamer, rx_streamer, tx_metadata, rx_metadata = setup_streamers(usrp)
    tx_buffer, t = dc_chirp(
        chirp_ampl, chirp_bw, sampling_rate, chirp_duration, ret_time_samples=True, pad=True
    )
    if len(tx_buffer.shape) == 1:
        tx_buffer = tx_buffer.reshape(1, tx_buffer.size)

    # Loop through frequencies
    logger.info("Beginning acquistion loop")
    recv_data_list = []
    for freq_idx, frequency in enumerate(center_freqs):
        # Create tx and rx threads
        recv_data_buff = np.zeros(
            [
                rx_samples,
            ],
            dtype=np.complex64,
        )
        recv_data_for_freq = np.zeros(
            [
                num_averages,
                rx_samples,
            ],
            dtype=np.complex64,
        )

        tune_center_freq(usrp, frequency)

        logger.info(
            f"Acquiring data at {frequency/1e9}GHz: center freq {freq_idx+1}/{num_freqs}..."
        )
        for jj in range(num_averages):
            rx_thread = threading.Thread(
                target=rx_worker, args=(usrp, rx_streamer, rx_metadata, recv_data_buff, verbose)
            )
            tx_thread = threading.Thread(
                target=tx_worker, args=(tx_streamer, tx_metadata, tx_buffer, verbose)
            )
            rx_thread.setName("rx_stream")
            tx_thread.setName("tx_stream")
            
            rx_thread.start()
            tx_thread.start()
            rx_thread.join()
            tx_thread.join()
            recv_data_for_freq[jj, :] = recv_data_buff 
            
        # @todo: align and sum signals before saving?
        recv_data_list.append(recv_data_for_freq)

    logger.info("Acquisition complete!")

    if len(output_filename):
        logger.info("Saving to file...")
        savemat(output_filename, {"data": recv_data_list})
        logger.info(f"Data written to {output_filename}")

    if plot_data:
        if verbose:
            logger.info("Plotting received data...")

        legend_text = ["Tx"]
        time_vec_rx = 1 / sampling_rate * np.arange(0, len(recv_data_list[0][0,:]))
        plt.figure()
        plt.plot(t * 1e6, np.real(tx_buffer[0, :]))
        for ii in range(num_freqs):
            for jj in range(num_averages):
                plt.plot(time_vec_rx * 1e6, np.real(recv_data_list[ii][jj,:]))
                legend_text.append(f"Rx {ii+1}, rep{jj+1}")
        plt.title("Baseband signals")
        plt.xlabel("Time [us]")
        plt.legend(legend_text)
        plt.grid(True)

        # Frequency-domain data
        # tx_fd = np.fft.fft(tx_buffer[0, :])
        # freqs_tx = np.fft.fftfreq(len(tx_fd), d=t[1] - t[0])
        # freqs_rx = np.fft.fftfreq(
        #     len(recv_data_list[0]), d=time_vec_rx[1] - time_vec_rx[0]
        # )
        # plt.figure()
        # plt.plot(freqs_tx / 1e6, 20 * np.log10(np.abs(tx_fd / len(tx_fd))))
        # for ii in range(num_freqs):
        #     rx_fd = np.fft.fft(recv_data_list[ii])
        #     plt.plot(freqs_rx / 1e6, 20 * np.log10(np.abs(rx_fd / len(rx_fd))), "--")

        # plt.xlabel("Frequency [MHz]")
        # plt.ylabel("Magnitude [dB]")
        # plt.legend(legend_text)
        # plt.ylim(-80, -10)
        # plt.grid(True)
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
