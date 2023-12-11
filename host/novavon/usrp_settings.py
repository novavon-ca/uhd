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
    usrp = uhd.usrp.MultiUSRP("num_recv_frames=512") #recv_frame_size=512, 

    # Always select the subdevice first, the channel mapping affects the other settings
    if len(args["channel_list"]) > 1:
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))
        usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))

    if verbose:
        logger.info("Using Device: %s", usrp.get_pp_string())

    # Set the reference clock
    if not setup_ref(usrp, clock_ref, logger):
        return False

    # Set the PPS source
    usrp.set_time_source(pps)

    # At this point, we can assume the device has valid and locked clock and PPS

    # Tx settings
    for chan in args["channel_list"]:
        usrp.set_tx_rate(args["tx_rate"],chan)
        usrp.set_tx_gain(args["tx_gain"],chan)
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(args["center_freq"]), chan)
        usrp.set_tx_antenna("TX/RX", chan)
        usrp.set_tx_bandwidth(args["tx_rate"], chan)

    # Rx settings
    for chan in args["channel_list"]:
        usrp.set_rx_rate(args["rx_rate"], chan)
        usrp.set_rx_gain(args["rx_gain"], chan)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args["center_freq"]), chan)
        usrp.set_rx_bandwidth(args["rx_rate"], chan)
        usrp.set_rx_antenna("RX2", chan)  # "RX2" or "TX/RX"
        if args["rx_auto_gain"]:
            logger.info(f"Using rx auto gain on channel{chan+1}")
            usrp.set_rx_agc(True, chan)

    # Read back settings
    if verbose:
        logger.info("Actual TX1 Freq: %f MHz...", usrp.get_tx_freq(0) / 1e6)
        logger.info("Actual TX2 Freq: %f MHz...", usrp.get_tx_freq(1) / 1e6)
        logger.info("Actual TX1 Gain: %f dB...", usrp.get_tx_gain(0))
        logger.info("Actual TX2 Gain: %f dB...", usrp.get_tx_gain(1))
        logger.info("Actual TX1 Bandwidth: %f MHz...", usrp.get_tx_bandwidth(0) / 1e6)
        logger.info("Actual TX2 Bandwidth: %f MHz...", usrp.get_tx_bandwidth(1) / 1e6)

        logger.info("Actual RX1 Freq: %f MHz...", usrp.get_rx_freq(0) / 1e6)
        logger.info("Actual RX2 Freq: %f MHz...", usrp.get_rx_freq(1) / 1e6)
        logger.info("Actual RX1 Gain: %f dB...", usrp.get_rx_gain(0))
        logger.info("Actual RX2 Gain: %f dB...", usrp.get_rx_gain(1))
        logger.info("Actual RX1 Bandwidth: %f MHz...", usrp.get_rx_bandwidth(0) / 1e6)
        logger.info("Actual RX2 Bandwidth: %f MHz...", usrp.get_rx_bandwidth(1) / 1e6)
        

    if len(args["channel_list"]) > 1:
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    else:
        usrp.set_time_now(uhd.types.TimeSpec(0.0))
    if verbose:
        logger.info("Set device timestamp to 0")

    return usrp


def setup_streamers(usrp):
    cpu_sample_mode = "fc32"
    otw_sample_mode = "sc8" # receive 8-bit data over-the-wire (can be 8 ,12(?) or 16)
    st_args = uhd.usrp.StreamArgs(cpu_sample_mode, otw_sample_mode)
    st_args.channels = [0, 1]
    tx_streamer = usrp.get_tx_stream(st_args)

    rx_streamer = usrp.get_rx_stream(st_args)

    rx_metadata = uhd.types.RXMetadata()
    tx_metadata = uhd.types.TXMetadata()
    tx_metadata.has_time_spec = False
    tx_metadata.start_of_burst = True
    tx_metadata.end_of_burst = False

    return tx_streamer, rx_streamer, tx_metadata, rx_metadata
