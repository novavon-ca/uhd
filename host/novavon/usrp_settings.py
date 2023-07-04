import uhd
import logging
from datetime import datetime, timedelta
import time
from typing import Any
import numpy as np


def setup_ref(usrp, ref, logger):
    """Setup the reference clock"""

    usrp.set_clock_source(ref)

    # Lock onto clock signals
    clock_timeout = 1000  # 1000mS timeout for external clock locking
    if ref != "internal":
        logger.debug("Now confirming lock on clock signals...")
        end_time = datetime.now() + timedelta(milliseconds=clock_timeout)
        is_locked = usrp.get_mboard_sensor("ref_locked", 0)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(1e-3)
            is_locked = usrp.get_mboard_sensor("ref_locked", 0)
        if not is_locked:
            logger.error("Unable to confirm clock signal locked on board %d", 0)
            return False
    return True


def usrp_setup(args, logger, verbose=False):
    """
    Sets up USRP device according to user-defined args
    returns: MultiUSRP object
    """

    # TODO: later this can come from a settings JSON file
    clock_ref = "internal"
    pps = "internal"

    # Create usrp device
    usrp = uhd.usrp.MultiUSRP()
    if verbose:
        logger.info("Using Device: %s", usrp.get_pp_string())

    # Set the reference clock
    if not setup_ref(usrp, clock_ref, logger):
        return False

    # Set the PPS source
    usrp.set_time_source(pps)

    # At this point, we can assume the device has valid and locked clock and PPS

    # Tx settings
    usrp.set_tx_rate(args["tx_rate"])
    usrp.set_tx_gain(args["tx_gain"], 0)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(args["center_freq"]), 0)
    usrp.set_tx_antenna("TX/RX", 0)

    # Rx settings
    usrp.set_rx_rate(args["rx_rate"])
    usrp.set_rx_gain(args["rx_gain"], 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args["center_freq"]), 0)
    usrp.set_rx_antenna("RX2", 0)  # "RX2" or "TX/RX"
    if args["rx_auto_gain"]:
        logger.info('Using rx auto gain')
        usrp.set_rx_agc(True, 0)

    # Read back settings
    if verbose:
        logger.info("Actual TX Freq: %f MHz...", usrp.get_tx_freq(0) / 1e6)
        logger.info("Actual TX Gain: %f dB...", usrp.get_tx_gain(0))
        logger.info("Actual TX Bandwidth: %f MHz...", usrp.get_tx_bandwidth(0)/ 1e6)

        logger.info("Actual RX Freq: %f MHz...", usrp.get_rx_freq(0) / 1e6)
        logger.info("Actual RX Gain: %f dB...", usrp.get_rx_gain(0))
        logger.info("Actual RX Bandwidth: %f MHz...", usrp.get_rx_bandwidth(0)/ 1e6)

    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    if verbose:
        logger.info("Set device timestamp to 0")

    return usrp


def tx_setup(usrp):
    cpu_sample_mode = "fc32"
    otw_sample_mode = "sc16"
    st_args = uhd.usrp.StreamArgs(cpu_sample_mode, otw_sample_mode)
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)

    max_samps_per_packet = tx_streamer.get_max_num_samps()
    tx_buf = np.zeros((1, max_samps_per_packet), dtype=np.complex64)

    metadata = uhd.types.TXMetadata()
    metadata.has_time_spec = False
    metadata.start_of_burst = True
    metadata.end_of_burst = False

    return tx_streamer, tx_buf, metadata


def rx_setup(usrp):
    cpu_sample_mode = "fc32"
    otw_sample_mode = "sc16"
    st_args = uhd.usrp.StreamArgs(cpu_sample_mode, otw_sample_mode)
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)

    num_samples_per_packet: int = int(rx_streamer.get_max_num_samps())
    metadata = uhd.types.RXMetadata()

    recv_buf: np.ndarray = np.empty((1, num_samples_per_packet), dtype=np.complex64)

    return rx_streamer, recv_buf, metadata
