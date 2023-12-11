import uhd
import logging
import numpy as np
import sys
import time
import threading
from typing import Any, List
import matplotlib.pyplot as plt
from scipy.io import savemat

from utilities import LogFormatter
# from waveforms import dc_chirp
# from usrp_settings import usrp_setup, setup_streamers

def setup_device(samp_rate, logger, verbose=False):
    device_args = "type=b200,recv_frame_size=1024"
    subdev = "A:A A:B"
    clock_source = "internal"
    rx_bw = samp_rate
    tx_bw = samp_rate
    mcr = samp_rate
    channel_list = [0,1]
    cpu_sample_mode = "fc32"
    otw_sample_mode = "sc8" # receive 8-bit data over-the-wire (can be 8 ,12(?) or 16)

    # create the USRP device
    usrp = uhd.usrp.MultiUSRP()
    #  usrp = uhd.usrp.MultiUSRP(device_args)

    # set clocks
    usrp.set_clock_source(clock_source)
    usrp.set_master_clock_rate(mcr)

    # select subdevice
    subdev_spec = uhd.usrp.SubdevSpec(subdev)
    usrp.set_rx_subdev_spec(subdev_spec)
    usrp.set_tx_subdev_spec(subdev_spec)
    if verbose:
        logger.info("Using Device: %s", usrp.get_pp_string())

    # set sample rate for all channels
    usrp.set_tx_rate(samp_rate)
    usrp.set_rx_rate(samp_rate)

    # set bandwidth
    usrp.set_tx_bandwidth(tx_bw, 0)
    usrp.set_tx_bandwidth(tx_bw, 1)
    usrp.set_rx_bandwidth(rx_bw, 0)
    usrp.set_rx_bandwidth(rx_bw, 1)

    # Read back settings
    if verbose:
        logger.info("Actual TX1 Rate: %f MHz...", usrp.get_tx_rate(0) / 1e6)
        logger.info("Actual TX2 Rate: %f MHz...", usrp.get_tx_rate(1) / 1e6)
        logger.info("Actual TX1 Bandwidth: %f MHz...", usrp.get_tx_bandwidth(0) / 1e6)
        logger.info("Actual TX2 Bandwidth: %f MHz...", usrp.get_tx_bandwidth(1) / 1e6)

        logger.info("Actual RX1 Rate: %f MHz...", usrp.get_rx_rate(0) / 1e6)
        logger.info("Actual RX2 Rate: %f MHz...", usrp.get_rx_rate(1) / 1e6)
        logger.info("Actual RX1 Bandwidth: %f MHz...", usrp.get_rx_bandwidth(0) / 1e6)
        logger.info("Actual RX2 Bandwidth: %f MHz...", usrp.get_rx_bandwidth(1) / 1e6)

    # create stream args and tx streamer
    st_args = uhd.usrp.StreamArgs(cpu_sample_mode, otw_sample_mode)
    st_args.channels = channel_list
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)

    return usrp, tx_streamer, rx_streamer

def tx_worker(usrp, tx_streamer, tx_data, tx_md):
    total_samps = 2000
    num_acc_samps = 0
    num_tx_samps = 0
    num_channels = tx_streamer.get_num_channels()

    tx_md.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + TX_DELAY)

    while num_acc_samps < total_samps:
        num_tx_samps += tx_streamer.send(tx_data, tx_md) * num_channels
        num_acc_samps += min(total_samps - num_acc_samps, tx_streamer.get_max_num_samps())

def tune_center_freq(usrp, target_freq):
    global STATE
    tx_gain = 60
    rx_gain = 45
    tx_chan_idx = 1

    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(target_freq), 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(target_freq), 1)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(target_freq), tx_chan_idx)

    # set freq-dependent gain here
    usrp.set_tx_gain(tx_gain, tx_chan_idx)
    usrp.set_rx_gain(rx_gain, 0)
    usrp.set_rx_gain(rx_gain, 1)
    
    # wait until LOs are locked
    while not (usrp.get_rx_sensor("lo_locked", 0).to_bool() and usrp.get_tx_sensor("lo_locked").to_bool()):
        pass

    STATE = "RX_AND_SAVE"
    return

def rx_and_save(usrp, rx_streamer, rx_buffer, rx_md, num_samps, current_freq):
    global STATE

    if current_freq > END_FREQ:
        STATE = "DONE"
        return

    logger.info("Current freq: %d MHz", current_freq/1e6)
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = False
    stream_cmd.time_spec = usrp.get_time_now() + uhd.types.TimeSpec(0.01) #uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + TX_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)

    rx = rx_streamer.recv(rx_buffer, rx_md)
    # logger.info(rx_md)
    STATE = "TUNE_FREQ"

def generate_tx_data(samp_rate):
    plot_data = False
    chan_list = (0,1)
    chirp_len_samples = 85000/2
    bw = 16e6
    n = np.arange(0, chirp_len_samples-1) - chirp_len_samples/2
    t = n/samp_rate
    ampl = 0.8

    chirp = ampl * np.array(np.exp(1j * np.pi * 0.5 * (bw/t[-1])*(t**2)), dtype=np.complex64)
    if plot_data:
        plt.figure()
        plt.subplot(211)
        plt.plot(t,np.real(chirp))
        plt.plot(t,np.imag(chirp))
        plt.xlabel('Time [s]')
        plt.title('Chirp source')

    pad_len = 4096
    chirp = np.pad(chirp, pad_len, 'constant', constant_values=(0))
    
    if plot_data:
        plt.subplot(212)
        plt.plot(np.fft.fftfreq(len(chirp), d=(t[1]-t[0]))/1e6,20 * np.log10(np.abs(np.fft.fft(chirp))))
        plt.xlabel('Freq [MHz]')
        plt.ylabel('Magnitude [dB]')
    
    # tile data for 2 channels
    tx_data = np.tile(chirp, (1,1))
    tx_data = np.tile(tx_data[0], (len(chan_list),1))

    # create metadata
    tx_md = uhd.types.TXMetadata()
    tx_md.start_of_burst = True
    tx_md.end_of_burst = False
    tx_md.has_time_spec = True

    chirp_duration = chirp.size * samp_rate
    return tx_data, tx_md, chirp.size, chirp_duration


def main():
    global STATE
    STATE = "TUNE_FREQ"
    output_filename = 'sfcw_jan12_halfchirplen_switchTxDBoard_B'
    plot_data = True
    samp_rate = 16e6
    tx_gain = 60
    rx_gain = 45

    usrp, tx_streamer, rx_streamer = setup_device(samp_rate, tx_gain, rx_gain, logger, verbose=False)

    tx_data, tx_md, chirp_samples, chirp_duration = generate_tx_data(samp_rate)
    
    # create receive buffers
    num_rx_samps = chirp_samples * 3
    rx_buffer = np.zeros((2, num_rx_samps), dtype=np.complex64)
    rx_md = uhd.types.RXMetadata()
    rx_data_list = []

    gain_idx = 0
    freq_list = np.arange(START_FREQ, END_FREQ, FREQ_STEP)

    for f_idx, current_freq in enumerate(freq_list):
        rx_buffer = np.zeros((2, num_rx_samps), dtype=np.complex64)
        
        usrp.set_time_now(uhd.types.TimeSpec(0.0))
        tune_center_freq(usrp, current_freq)
        
        tx_thread = threading.Thread(target=tx_worker, args=(usrp, tx_streamer, tx_data, tx_md))
        rx_thread = threading.Thread(target=rx_and_save, args=(usrp, rx_streamer, rx_buffer, rx_md, num_rx_samps, current_freq))
        
        tx_thread.setName("tx_stream")
        rx_thread.setName("rx_stream")

        rx_thread.start()
        tx_thread.start()

        rx_thread.join()
        tx_thread.join()

        rx_data_list.append(rx_buffer)        
    
    # end = time.time()

    if output_filename:
        savemat(output_filename, {
            "data": rx_data_list, 
            'sampling_rate': samp_rate, 
            'frequencies': freq_list, 
            'chirp_duration': chirp_duration, 
            'chirp_bw':samp_rate, 
            'tx_gain': tx_gain, 
            'rx_gain': rx_gain
            })
        logger.info(f"Data written to {output_filename}")
    
    if plot_data:
        rx_data_list = np.array(rx_data_list)
        # print(rx_data_list.shape)
        
        plt.figure()
        plt.subplot(311)
        plt.title('F1')
        plt.plot(rx_data_list[0,0,:])
        plt.plot(rx_data_list[0,1,:])

        plt.subplot(312)
        plt.title('F2')
        plt.plot(rx_data_list[1,0,:])
        plt.plot(rx_data_list[1,1,:])

        plt.subplot(313)
        plt.title('F3')
        plt.plot(rx_data_list[20,0,:])
        plt.plot(rx_data_list[20,1,:])

        plt.show()





if __name__ == "__main__":
    START_FREQ = 800e6
    END_FREQ = 2e9
    FREQ_STEP = 16e6
    TX_DELAY = 0.012

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


