import uhd 
import logging
from datetime import datetime, timedelta
import time


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
    # Create usrp device
    usrp = uhd.usrp.MultiUSRP()
    if verbose:
        logger.info("Using Device: %s", usrp.get_pp_string())
    
    # Set the reference clock
    if not setup_ref(usrp, args["clock_ref"], logger):
        return False

    # Set the PPS source
    usrp.set_time_source(args["pps"])
    
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
    usrp.set_rx_antenna(args["rx_antenna"], 0)
    if args["rx_auto_gain"]:
        usrp.set_rx_agc(True, 0)

    # Read back settings
    if verbose:
        logger.info("Actual TX Freq: %f MHz...", usrp.get_tx_freq(0) / 1e6)
        logger.info("Actual TX Gain: %f dB...", usrp.get_tx_gain(0))
        logger.info("Actual TX Bandwidth: %f MHz...", usrp.get_tx_bandwidth(0))

        logger.info("Actual RX Freq: %f MHz...", usrp.get_rx_freq(0) / 1e6)
        logger.info("Actual RX Gain: %f dB...", usrp.get_rx_gain(0))
        logger.info("Actual RX Bandwidth: %f MHz...", usrp.get_rx_bandwidth(0))

    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    if verbose:
        logger.info("Set device timestamp to 0")

    return usrp