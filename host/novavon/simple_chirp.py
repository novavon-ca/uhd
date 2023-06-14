import uhd
import argparse
import logging
import numpy as np
import sys
import threading
from typing import Any, List

from scipy.io import savemat

from logs import LogFormatter
from waveforms import dc_chirp
from usrp_settings import usrp_setup

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


def rx_worker(usrp, rx_streamer, rx_statistics, rx_data):
    """Receive a fixed number of samples and store in rx_data"""

    # Make a receive buffer
    num_channels: int = int(rx_streamer.get_num_channels())
    num_samples_per_packet: int = 1000 #int(rx_streamer.get_max_num_samps())
    total_samples = len(rx_data)
    assert(num_samples_per_packet <= rx_streamer.get_max_num_samps())
    metadata = uhd.types.RXMetadata()
    recv_buffer: np.ndarray = np.empty((num_channels, num_samples_per_packet), dtype=np.complex64)

    # Craft and send the Stream Command
    # continuous capture:
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)

    # capture a fixed num of samples
    # @todo: probably want this option in future
    # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    # stream_cmd.num_samps = rx_num_samps

    stream_cmd.stream_now = True
    stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + RX_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)

    num_rx_samps: int = 0

    # Receive until we get the signal to stop
    # while not timer_elapsed_event.is_set():
    for ii in range(total_samples//num_samples_per_packet):
        try:
            rx = rx_streamer.recv(recv_buffer, metadata)
            rx_data[ii*num_samples_per_packet:(ii+1)*num_samples_per_packet] = recv_buffer[0]
            num_rx_samps += int(rx) * num_channels
        
        except RuntimeError as ex:
            logger.error("Runtime error in receive: %s", ex)
            return

    # Return the statistics to the main thread
    rx_statistics["num_rx_samps"] = num_rx_samps

    # After we get the signal to stop, issue a stop command
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))


def tx_worker(usrp, tx_streamer, tx_statistics, transmit_buffer):
    """Stream data stored in transmit_buffer"""
    
    assert(len(transmit_buffer) <= tx_streamer.get_max_num_samps())

    # Make a transmit buffer
    num_channels = 1
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = False
    metadata.has_time_spec = False
    # metadata.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + RX_DELAY)

    # Transmit a fixed number of samples
    num_cycles = 1
    total_num_samps = np.size(transmit_buffer, 0) * num_cycles
    num_tx_samps = 0
    num_acc_samps = 0
    while num_acc_samps < total_num_samps:
        num_tx_samps += tx_streamer.send(transmit_buffer, metadata) * num_channels
        num_acc_samps += min(total_num_samps - num_acc_samps,
                                tx_streamer.get_max_num_samps())
        # metadata.has_time_spec = False
        metadata.start_of_burst = False

    tx_statistics["num_tx_samps"] = num_tx_samps

    # Send a mini EOB packet
    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), metadata)


def print_statistics(rx_statistics, tx_statistics):
    """Print TRX statistics in a formatted block"""
    
    statistics_msg = """
    Num received samples:     {}
    Num transmitted samples:  {}
    """.format(
        rx_statistics.get("num_rx_samps", 0),
        tx_statistics.get("num_tx_samps", 0))
    logger.info(statistics_msg)


def start_threads(usrp, args, tx_buf, rx_buf):
    # @TODO: make this able to take in more general tx and rx workers
    threads: List[threading.Thread] = []

    rx_statistics: dict[str, int] = {}
    st_args = uhd.usrp.StreamArgs(args["rx_cpu_sample_mode"], args["rx_otw_sample_mode"])
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)
    rx_thread = threading.Thread(target=rx_worker,
                                    args=(usrp, rx_streamer, rx_statistics, rx_buf))
    threads.append(rx_thread)
    rx_thread.setName("rx_stream")

    tx_statistics = {}
    st_args = uhd.usrp.StreamArgs(args["tx_cpu_sample_mode"], args["tx_otw_sample_mode"])
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    tx_thread = threading.Thread(target=tx_worker,args=(usrp, tx_streamer, tx_statistics, tx_buf))
    threads.append(tx_thread)
    tx_thread.setName("tx_stream")

    rx_thread.start()
    tx_thread.start()

    for thr in threads:
        thr.join()

    return tx_buf, rx_buf, tx_statistics, rx_statistics


def generate_output(args, tx_data, rx_data, tx_stats, rx_stats):

    if args.get("output_filename", ""):
        filename = args["output_filename"]
        if args["verbose"]:
            logger.info("Acquisition complete! Writing to file...")
        
        savemat(filename, {"data": rx_data})
        logger.info(f"Data written to {filename}")
    
    if args["plot_data"]:
        if args["verbose"]:
            logger.info("Plotting received data...")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.real(tx_data))
        plt.plot(np.real(rx_data))
        plt.legend(["Transmitted", "Received"])
        plt.show()
    
    if args["verbose"]:
        print_statistics(rx_stats, tx_stats)
    
    return


def main():
    success = False

    args: dict[str, Any] = {
        "clock_ref": "internal",
        "pps": "internal",
        "center_freq": 0.85e9,
        "sampling_rate": 20e6, # samples per second
        "chirp_bw": 5e6,
        "chirp_ampl": 0.3, #float between 0 and 1

        # "tx_rate": sampling_rate,
        "tx_samples": 2000,
        "tx_gain": 60, # [dB]
        "tx_cpu_sample_mode": "fc32",
        "tx_otw_sample_mode": "sc16",

        # "rx_rate": sampling_rate,
        "rx_samples": 50000,
        "rx_antenna": "RX2", # "RX2" or "TX/RX"
        "rx_gain": 50,  # [dB]
        "rx_cpu_sample_mode": "fc32",
        "rx_otw_sample_mode": "sc16",
        "rx_auto_gain": False,

        "output_filename": "TEST.mat", # set to empty string to not save data to file
        "plot_data": True,
        "verbose": False,
    }

    args.update({"tx_rate": args["sampling_rate"], "rx_rate": args["sampling_rate"]})
    verbose = args["verbose"]

    # TODO: implement arg checking, command-line args
    # args = argparse()
    # success, err_msg = validate_args(args)
    # if not success:
    #     logging.error(err_msg)

    usrp = usrp_setup(args, logger, verbose)

    rx_buffer = np.zeros(args["rx_samples"], dtype=np.complex64)
    tx_buffer = dc_chirp(args["chirp_ampl"], args["chirp_bw"], args["sampling_rate"], args['tx_samples'])
    tx_dat, rx_dat, tx_stats, rx_stats = start_threads(usrp, args, tx_buffer, rx_buffer)
    generate_output(args, tx_dat, rx_dat, tx_stats, rx_stats)

    return success


if __name__ == "__main__":
    RX_DELAY = 0.01 # offset delay between transmitting and receiving @TODO: put this into args?

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)
    formatter = LogFormatter(fmt='[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s')
    console.setFormatter(formatter)

    sys.exit(not main())
