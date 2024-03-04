#!/usr/bin/env python
#
# Copyright 2018 Ettus Research, a National Instruments Company
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Benchmark rate using Python API
"""

import argparse
from datetime import datetime, timedelta
import sys
import time
import threading
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
import uhd
from waveforms import dc_chirp


CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit

def parse_args():
    """Parse the command line arguments"""
    description = """UHD Benchmark Rate (Python API)

        Utility to stress test a USRP device.

        Specify --rx_rate for a receive-only test.
        Specify --tx_rate for a transmit-only test.
        Specify both options for a full-duplex test.
        """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=description)
    parser.add_argument("-a", "--args", default="", type=str, help="single uhd device address args")
    parser.add_argument("-d", "--duration", default=0.05, type=float,
                        help="duration for the test in seconds")
    parser.add_argument("--rx_subdev", default="A:A A:B", type=str, help="specify the device subdev for RX")
    parser.add_argument("--tx_subdev", default="A:A A:B", type=str, help="specify the device subdev for TX")
    parser.add_argument("--rx_rate", default=20e6, type=float, help="specify to perform a RX rate test (sps)")
    parser.add_argument("--tx_rate", default=20e6, type=float, help="specify to perform a TX rate test (sps)")
    parser.add_argument("--rx_otw", type=str, default="sc8",
                        help="specify the over-the-wire sample mode for RX")
    parser.add_argument("--tx_otw", type=str, default="sc8",
                        help="specify the over-the-wire sample mode for TX")
    parser.add_argument("--rx_cpu", type=str, default="fc32",
                        help="specify the host/cpu sample mode for RX")
    parser.add_argument("--tx_cpu", type=str, default="fc32",
                        help="specify the host/cpu sample mode for TX")
    parser.add_argument("--ref", type=str, default="internal",
                        help="clock reference (internal, external, mimo, gpsdo)")
    parser.add_argument("--pps", type=str, default="internal", 
                        help="PPS source (internal, external, mimo, gpsdo)")
    parser.add_argument("--random", action="store_true", default=False,
                        help="Run with random values of samples in send() and recv() to stress-test"
                             " the I/O.")
    parser.add_argument("-c", "--channels", default=[0,1], nargs="+", type=int,
                        help="which channel(s) to use (specify \"0\", \"1\", \"0 1\", etc)")
    parser.add_argument("--rx_channels", nargs="+", type=int,
                        help="which RX channel(s) to use (specify \"0\", \"1\", \"0 1\", etc)")
    parser.add_argument("--tx_channels", nargs="+", type=int,
                        help="which TX channel(s) to use (specify \"0\", \"1\", \"0 1\", etc)")
    return parser.parse_args()


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


def setup_ref(usrp, ref, num_mboards):
    """Setup the reference clock"""
    if ref == "mimo":
        if num_mboards != 2:
            logger.error("ref = \"mimo\" implies 2 motherboards; "
                          "your system has %d boards", num_mboards)
            return False
        usrp.set_clock_source("mimo", 1)
    else:
        usrp.set_clock_source(ref)

    # Lock onto clock signals for all mboards
    if ref != "internal":
        logger.debug("Now confirming lock on clock signals...")
        end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
        for i in range(num_mboards):
            if ref == "mimo" and i == 0:
                continue
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
            while (not is_locked) and (datetime.now() < end_time):
                time.sleep(1e-3)
                is_locked = usrp.get_mboard_sensor("ref_locked", i)
            if not is_locked:
                logger.error("Unable to confirm clock signal locked on board %d", i)
                return False
    return True


def setup_pps(usrp, pps, num_mboards):
    """Setup the PPS source"""
    if pps == "mimo":
        if num_mboards != 2:
            logger.error("ref = \"mimo\" implies 2 motherboards; "
                          "your system has %d boards", num_mboards)
            return False
        # make mboard 1 a slave over the MIMO Cable
        usrp.set_time_source("mimo", 1)
    else:
        usrp.set_time_source(pps)
    return True


def check_channels(usrp, args):
    """Check that the device has sufficient RX and TX channels available"""
    # Check RX channels
    if args.rx_rate:
        if args.rx_channels:
            rx_channels = args.rx_channels
        else:
            rx_channels = args.channels
        # Check that each channel specified is less than the number of total number of rx channels
        # the device can support
        dev_rx_channels = usrp.get_rx_num_channels()
        if not all(map((lambda chan: chan < dev_rx_channels), rx_channels)):
            logger.error("Invalid RX channel(s) specified.")
            return [], []
    else:
        rx_channels = []
    # Check TX channels
    if args.tx_rate:
        if args.tx_channels:
            tx_channels = args.tx_channels
        else:
            tx_channels = args.channels
        # Check that each channel specified is less than the number of total number of tx channels
        # the device can support
        dev_tx_channels = usrp.get_tx_num_channels()
        if not all(map((lambda chan: chan < dev_tx_channels), tx_channels)):
            logger.error("Invalid TX channel(s) specified.")
            return [], []
    else:
        tx_channels = []
    return rx_channels, tx_channels


def benchmark_rx_rate(usrp, rx_streamer, random, timer_elapsed_event, rx_statistics, rx_data):
    """Benchmark the receive chain"""
    logger.info("Testing receive rate {:.3f} Msps on {:d} channels".format(
        usrp.get_rx_rate()/1e6, rx_streamer.get_num_channels()))

    # Make a receive buffer
    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    # TODO: The C++ code uses rx_cpu type here. Do we want to use that to set dtype?
    recv_buffer = np.empty((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # To estimate the number of dropped samples in an overflow situation, we need the following
    # On the first overflow, set had_an_overflow and record the time
    # On the next ERROR_CODE_NONE, calculate how long its been since the recorded time, and use the
    #   tick rate to estimate the number of dropped samples. Also, reset the tracking variables
    had_an_overflow = False
    last_overflow = uhd.types.TimeSpec(0)
    # Setup the statistic counters
    num_rx_samps = 0
    num_rx_dropped = 0
    num_rx_overruns = 0
    num_rx_seqerr = 0
    num_rx_timeouts = 0
    num_rx_late = 0

    rate = usrp.get_rx_rate()
    ii = 0
    total_samples: int = np.size(rx_data, 1)

    rx_dict = {'data_re': [],
               'data_im': [],
               'packet_start': [],
               'packet_size': [],
               'error_code': []}
    sample_idx = 0

    logger.info("Rx max samps per packet: %d", max_samps_per_packet)

    # Craft and send the Stream Command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = (num_channels == 1)
    # start_time = usrp.get_time_now().get_real_secs() + INIT_DELAY
    # logger.info("Rx stream cmd issued at %s", start_time)
    stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # Receive until we get the signal to stop
    while not timer_elapsed_event.is_set():
        if random:
            stream_cmd.num_samps = np.random.randint(1, max_samps_per_packet+1, dtype=int)
            rx_streamer.issue_stream_cmd(stream_cmd)
        
        try:
            samps = rx_streamer.recv(recv_buffer, metadata)
            num_rx_samps += samps * num_channels
            rx_data[:, ii : ii+max_samps_per_packet] = recv_buffer[:, 0:max_samps_per_packet]
            if samps:
                rx_dict['data_re'].append(np.real((recv_buffer[:, 0:samps])).tolist())
                rx_dict['data_im'].append(np.imag((recv_buffer[:, 0:samps])).tolist())
                rx_dict["packet_start"].append(metadata.time_spec.get_real_secs())
                rx_dict["packet_size"].append(samps)
                
                sample_idx += num_rx_samps

        except RuntimeError as ex:
            logger.error("Runtime error in receive: %s", ex)
            return

        # logger.info(metadata)

        # Handle the error codes
        if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
            # Reset the overflow flag
            if samps:
                rx_dict["error_code"].append('NONE')
            if had_an_overflow:
                had_an_overflow = False
                num_rx_dropped += uhd.types.TimeSpec(
                    metadata.time_spec.get_real_secs() - last_overflow.get_real_secs()
                ).to_ticks(rate)
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
            if samps:
                rx_dict["error_code"].append('OVERFLOW')
            had_an_overflow = True
            last_overflow = metadata.time_spec
            # If we had a sequence error, record it
            if metadata.out_of_sequence:
                num_rx_seqerr += 1
                logger.warning("Detected RX sequence error.")
            # Otherwise just count the overrun
            else:
                num_rx_overruns += 1
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.late:
            if samps:
                rx_dict["error_code"].append('LATE CMD')
            logger.warning("Receiver error: %s, restarting streaming...", metadata.strerror())
            num_rx_late += 1
            # Radio core will be in the idle state. Issue stream command to restart streaming.
            stream_cmd.time_spec = uhd.types.TimeSpec(
                usrp.get_time_now().get_real_secs() + INIT_DELAY)
            stream_cmd.stream_now = (num_channels == 1)
            rx_streamer.issue_stream_cmd(stream_cmd)
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
            if samps:
                rx_dict["error_code"].append('TIMEOUT')
            logger.warning("Receiver error: %s, continuing...", metadata.strerror())
            num_rx_timeouts += 1
        else:
            if samps:
                rx_dict["error_code"].append(f'OTHER: {metadata.strerror()}')
            logger.error("Receiver error: %s", metadata.strerror())
            logger.error("Unexpected error on receive, continuing...")
        
        ii = ii + 1
        if (ii >= (total_samples // max_samps_per_packet)):
            logger.info(f"Ending acquisition after {ii+1} packets")
            break
            # logger.info(f"Overwriting data after {ii+1} packets")
            # ii = 0
    
    # After we get the signal to stop, issue a stop command
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    # Return the statistics to the main thread
    rx_statistics["num_rx_samps"] = num_rx_samps
    rx_statistics["num_rx_dropped"] = num_rx_dropped
    rx_statistics["num_rx_overruns"] = num_rx_overruns
    rx_statistics["num_rx_seqerr"] = num_rx_seqerr
    rx_statistics["num_rx_timeouts"] = num_rx_timeouts
    rx_statistics["num_rx_late"] = num_rx_late
    rx_statistics["packet_data_re"] = rx_dict["data_re"]
    rx_statistics["packet_data_im"] = rx_dict["data_im"]
    rx_statistics["packet_start"] = rx_dict["packet_start"]
    rx_statistics["packet_size"] = rx_dict["packet_size"]
    rx_statistics["error_code"] = rx_dict["error_code"]

    
    

def benchmark_tx_rate(usrp, tx_streamer, random, timer_elapsed_event, tx_statistics, transmit_buffer):
    """Benchmark the transmit chain"""
    logger.info("Testing transmit rate %.3f Msps on %d channels",
                 usrp.get_tx_rate() / 1e6, tx_streamer.get_num_channels())

    # Make a transmit buffer
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()
    # logger.info("Tx max samps per packet: %d", max_samps_per_packet)
    # TODO: The C++ code uses rx_cpu type here. Do we want to use that to set dtype?
    
    # transmit_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.TXMetadata()
    # start_time = usrp.get_time_now().get_real_secs() + INIT_DELAY
    # logger.info("Tx send cmd issued at %s", start_time)
    metadata.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
    metadata.has_time_spec = bool(num_channels)

    # Setup the statistic counters
    tx_statistics["packet_start"] = []
    tx_statistics["packet_size"] = []
    num_tx_samps = 0
    num_tx_samps_now = 0
    # TODO: The C++ has a single randomly sized packet sent here, then the thread returns
    num_timeouts_tx = 0
    # Transmit until we get the signal to stop
    if random:
        while not timer_elapsed_event.is_set():
            total_num_samps = np.random.randint(1, max_samps_per_packet + 1, dtype=int)
            num_acc_samps = 0
            while num_acc_samps < total_num_samps:
                num_tx_samps += tx_streamer.send(
                    transmit_buffer, metadata) * num_channels
                num_acc_samps += min(total_num_samps - num_acc_samps,
                                     tx_streamer.get_max_num_samps())
    else:
        while not timer_elapsed_event.is_set():
            try:
                num_tx_samps_now = tx_streamer.send(transmit_buffer, metadata) * num_channels
                num_tx_samps += num_tx_samps_now
                if num_tx_samps_now == 0:
                    num_timeouts_tx += 1
                    if (num_timeouts_tx % 10000) == 1:
                        logger.warning("Tx timeouts: %d", num_timeouts_tx)
                metadata.has_time_spec = False
                # metadata.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)

            except RuntimeError as ex:
                logger.error("Runtime error in transmit: %s", ex)
                return

            tx_statistics["packet_start"].append(usrp.get_time_now().get_real_secs())
            # tx_statistics["packet_size"].append(num_tx_samps_now)

    tx_statistics["num_tx_samps"] = num_tx_samps

    # Send a mini EOB packet
    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), metadata)


def benchmark_tx_rate_async_helper(tx_streamer, timer_elapsed_event, tx_async_statistics):
    """Receive and process the asynchronous TX messages"""
    async_metadata = uhd.types.TXAsyncMetadata()

    # Setup the statistic counters
    num_tx_seqerr = 0
    num_tx_underrun = 0
    num_tx_timeouts = 0  # TODO: Not populated yet

    try:
        while not timer_elapsed_event.is_set():
            # Receive the async metadata
            if not tx_streamer.recv_async_msg(async_metadata, 0.1):
                continue

            # Handle the error codes
            if async_metadata.event_code == uhd.types.TXMetadataEventCode.burst_ack:
                return
            elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow) or
                  (async_metadata.event_code == uhd.types.TXMetadataEventCode.underflow_in_packet)):
                num_tx_underrun += 1
            elif ((async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error) or
                  (async_metadata.event_code == uhd.types.TXMetadataEventCode.seq_error_in_packet)):
                num_tx_seqerr += 1
            else:
                logger.warning("Unexpected event on async recv (%s), continuing.",
                                async_metadata.event_code)

    finally:
        # Write the statistics back
        tx_async_statistics["num_tx_seqerr"] = num_tx_seqerr
        tx_async_statistics["num_tx_underrun"] = num_tx_underrun
        tx_async_statistics["num_tx_timeouts"] = num_tx_timeouts


def print_statistics(rx_statistics, tx_statistics, tx_async_statistics):
    """Print TRX statistics in a formatted block"""
    # logger.debug("RX Statistics Dictionary: %s", rx_statistics)
    # logger.debug("TX Statistics Dictionary: %s", tx_statistics)
    # logger.debug("TX Async Statistics Dictionary: %s", tx_async_statistics)
    # Print the statistics
    statistics_msg = """Benchmark rate summary:
    Num received samples:     {}
    Num dropped samples:      {}
    Num overruns detected:    {}
    Num transmitted samples:  {}
    Num sequence errors (Tx): {}
    Num sequence errors (Rx): {}
    Num underruns detected:   {}
    Num late commands:        {}
    Num timeouts (Tx):        {}
    Num timeouts (Rx):        {}""".format(
        rx_statistics.get("num_rx_samps", 0),
        rx_statistics.get("num_rx_dropped", 0),
        rx_statistics.get("num_rx_overruns", 0),
        tx_statistics.get("num_tx_samps", 0),
        tx_async_statistics.get("num_tx_seqerr", 0),
        rx_statistics.get("num_rx_seqerr", 0),
        tx_async_statistics.get("num_tx_underrun", 0),
        rx_statistics.get("num_rx_late", 0),
        tx_async_statistics.get("num_tx_timeouts", 0),
        rx_statistics.get("num_rx_timeouts", 0))
    logger.info(statistics_msg)


def main():
    """Run the benchmarking tool"""
    args = parse_args()
    # Setup some argument parsing
    if not (args.rx_rate or args.tx_rate):
        logger.error("Please specify --rx_rate and/or --tx_rate")
        return False

    # Setup a usrp device
    usrp = uhd.usrp.MultiUSRP()
    if usrp.get_mboard_name() == "USRP1":
        logger.warning(
            "Benchmark results will be inaccurate on USRP1 due to insufficient features.")

    # Always select the subdevice first, the channel mapping affects the other settings
    if args.rx_subdev:
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(args.rx_subdev))
    if args.tx_subdev:
        usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(args.tx_subdev))

    logger.info("Using Device: %s", usrp.get_pp_string())

    # Set the reference clock
    if args.ref and not setup_ref(usrp, args.ref, usrp.get_num_mboards()):
        # If we wanted to set a reference clock and it failed, return
        return False

    # Set the PPS source
    if args.pps and not setup_pps(usrp, args.pps, usrp.get_num_mboards()):
        # If we wanted to set a PPS source and it failed, return
        return False
    # At this point, we can assume our device has valid and locked clock and PPS

    rx_channels, tx_channels = check_channels(usrp, args)
    if not rx_channels and not tx_channels:
        # If the check returned two empty channel lists, that means something went wrong
        return False
    logger.info("Selected %s RX channels and %s TX channels",
                 rx_channels if rx_channels else "no",
                 tx_channels if tx_channels else "no")

    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("RX2", 0)
    if len(rx_channels)>1:
        usrp.set_rx_antenna("RX2", 1)
    if len(tx_channels)>1:
        usrp.set_tx_antenna("TX/RX", 1)


    logger.info("Setting device timestamp to 0...")
    # If any of these conditions are met, we need to synchronize the channels
    if args.pps == "mimo" or args.ref == "mimo" or len(rx_channels) > 1 or len(tx_channels) > 1:
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    else:
        usrp.set_time_now(uhd.types.TimeSpec(0.0))

    threads = []
    # Make a signal for the threads to stop running
    quit_event = threading.Event()
    # Create a dictionary for the RX statistics
    # Note: we're going to use this without locks, so don't access it from the main thread until
    #       the worker has joined
    rx_statistics = {}
    
    # Spawn the receive test thread
    if args.rx_rate:
        for chan in rx_channels:
            usrp.set_rx_rate(args.rx_rate,chan)
            usrp.set_rx_gain(45,chan)            
        rx_data = np.zeros([len(rx_channels), int(4e5)], dtype=np.complex64)
        st_args = uhd.usrp.StreamArgs(args.rx_cpu, args.rx_otw)
        st_args.channels = rx_channels
        rx_streamer = usrp.get_rx_stream(st_args)
        rx_thread = threading.Thread(target=benchmark_rx_rate,
                                     args=(usrp, rx_streamer, args.random, quit_event,
                                           rx_statistics, rx_data))
        threads.append(rx_thread)
        rx_thread.start()
        rx_thread.setName("bmark_rx_stream")

    # Create a dictionary for the RX statistics
    # Note: we're going to use this without locks, so don't access it from the main thread until
    #       the worker has joined
    tx_statistics = {}
    tx_async_statistics = {}
    # Spawn the transmit test thread
    if args.tx_rate:
        for chan in tx_channels:
            usrp.set_tx_rate(args.tx_rate,chan)
            usrp.set_tx_gain(50,chan)
        tx_data = dc_chirp(0.3, 8e6, 20e6, 1.0e-4, pad=True)
        print(tx_data.shape)
        if len(tx_data.shape) == 1:
            tx_data = tx_data.reshape(1, tx_data.size)
        if len(tx_channels) > 1:
            tx_data = np.tile(tx_data, (len(tx_channels),1))

        st_args = uhd.usrp.StreamArgs(args.tx_cpu, args.tx_otw)
        st_args.channels = tx_channels
        tx_streamer = usrp.get_tx_stream(st_args)
        tx_thread = threading.Thread(target=benchmark_tx_rate,
                                     args=(usrp, tx_streamer, args.random, quit_event,
                                           tx_statistics, tx_data))
        threads.append(tx_thread)
        tx_thread.start()
        tx_thread.setName("bmark_tx_stream")

        tx_async_thread = threading.Thread(target=benchmark_tx_rate_async_helper,
                                           args=(tx_streamer, quit_event, tx_async_statistics))
        threads.append(tx_async_thread)
        tx_async_thread.start()
        tx_async_thread.setName("bmark_tx_helper")

    # Sleep for the required duration
    # If we have a multichannel test, add some time for initialization
    if len(rx_channels) > 1 or len(tx_channels) > 1:
        args.duration += INIT_DELAY
    time.sleep(args.duration)
    # Interrupt and join the threads
    logger.debug("Sending signal to stop!")
    quit_event.set()
    for thr in threads:
        thr.join()

    print_statistics(rx_statistics, tx_statistics, tx_async_statistics)

    with open('sample_benchmark_data_jan3.json', 'w') as f:
        json.dump({'packet_data_re': rx_statistics['packet_data_re'],
                   'packet_data_im': rx_statistics['packet_data_im'],
                   'packet_start': rx_statistics['packet_start'],
                   'packet_size': rx_statistics['packet_size'],
                   'error_code': rx_statistics["error_code"],
                   'sampling_rate': args.rx_rate,
                   'tx_packet_start': tx_statistics['packet_start'],
                   'tx_packet_size': tx_statistics['packet_size'],
                   }, f)

    # print(len(rx_statistics['packet_data_re'][0]),len(rx_statistics['packet_data_re'][0][0]))

    time_vec_rx = 1 / args.rx_rate * np.arange(0, np.size(rx_data,1))
    time_vec_tx = 1 / args.tx_rate * np.arange(0, np.size(tx_data,1))
    
    plt.figure()
    plt.plot(time_vec_tx * 1e6, np.real(tx_data[0,:]))
    plt.plot(time_vec_rx * 1e6, np.real(rx_data[0,:]))
    if len(rx_channels)>1:
        plt.plot(time_vec_rx * 1e6, np.real(rx_data[1,:]+0.1))
    plt.xlabel("Time [us]")
    plt.legend(["Tx1","Rx1", "Rx2"])
    plt.grid(True)

    plt.figure()
    num_packets = len(rx_statistics['packet_start'])
    for ii in range(num_packets):
        start = rx_statistics["packet_start"][ii]
        num = rx_statistics["packet_size"][ii]
        t = np.linspace(start, start + num/args.rx_rate, num=num)
        if rx_statistics["error_code"][ii] == "NONE":
            plt.plot(t, rx_statistics['packet_data_re'][ii][0], 'b')
            plt.plot(t, rx_statistics['packet_data_re'][ii][1], 'r')
        else:
            plt.plot(t, rx_statistics['packet_data_re'][ii][0], 'b--')
            plt.plot(t, rx_statistics['packet_data_re'][ii][1], 'r--')
    
    plt.figure()
    plt.plot( tx_statistics['packet_start'],list(range(len(tx_statistics['packet_start']))), 'o')
    plt.plot( rx_statistics['packet_start'],list(range(len(rx_statistics['packet_start']))), 'x')

    plt.show()

    return True


if __name__ == "__main__":
    # Setup the logger with our custom timestamp formatting
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)
    formatter = LogFormatter(fmt='[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s')
    console.setFormatter(formatter)

    # Vamos, vamos, vamos!
    sys.exit(not main())
