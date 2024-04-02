import uhd
import numpy as np
import sys
import time
import threading
import logging
import os
from scipy.io import savemat

from sfcw import setup_device, tx_worker, get_gain_adjustments, tune_center_freq, rx_and_save, generate_tx_data
from utilities import LogFormatter


def main():
    # ----- Initialization ----- #
    # USRP things
    usrp, tx_streamer, rx_streamer = setup_device(SAMP_RATE, logger, verbose=False)
    tx_data, tx_md, chirp_samples, chirp_duration = generate_tx_data(SAMP_RATE, CHIRP_BW)
    gain_ch0, gain_ch1 = get_gain_adjustments("/home/novavon/Desktop/gain_tables.json", START_FREQ, CHIRP_BW)

    # DAQ variables
    num_rx_samps = chirp_samples * 3
    rx_md = uhd.types.RXMetadata()
    freq_list = np.arange(START_FREQ, END_FREQ, FREQ_STEP)
    assert(len(gain_ch1)>=len(freq_list))

    # Directory to save data
    now: time.struct_time = time.localtime()
    save_dir: str = f"/home/client/sdrdata/{now.tm_year}-{now.tm_mon}-{now.tm_mday}/"
    if not os.path.exists(save_dir):
        os.mkdirs(save_dir)
    
    print('Data will be saved to the following directory: ', save_dir)

    # Write header file with acqusition parameters
    savemat(
        'header.mat', {
        'start_time': f"{now.tm_hour}:{now.tm_min}:{now.tm_sec}",
        'sampling_rate': SAMP_RATE, 
        'frequencies': freq_list, 
        'chirp_duration': chirp_duration, 
        'chirp_bw':CHIRP_BW
    })
    start_secs: float = time.time()
    secs_since_start = 0

    # ----- Acquisition Loop ----- #
    while (secs_since_start < QUIT_AFTER_SECONDS):
        rx_data_list = [] # stores data for all frequencies at this iteration of the main loop

        # Frequency hopping loop
        for f_idx, current_freq in enumerate(freq_list):
            rx_buffer = np.zeros((2, num_rx_samps), dtype=np.complex64)

            tune_center_freq(usrp, current_freq, [gain_ch0[f_idx], gain_ch1[f_idx]])
            tx_thread = threading.Thread(target=tx_worker, args=(usrp, tx_streamer, tx_data, tx_md), name="tx_stream")
            rx_thread = threading.Thread(target=rx_and_save, args=(usrp, rx_streamer, rx_buffer, rx_md, num_rx_samps, current_freq), name="rx_stream")
            
            rx_thread.start()
            tx_thread.start()
            rx_thread.join()
            tx_thread.join()

            rx_data_list.append(rx_buffer) 

        # Write contents of rx_data_list to file - filename = # seconds since program started
        secs_since_start = time.time() - start_secs
        savemat(save_dir + f"{secs_since_start}.mat", {"data": rx_data_list})
    
    logger.info('Acquisition timer finished')


if __name__ == "__main__":
    START_FREQ = 1010e6
    END_FREQ = 2010e6
    FREQ_STEP = 15e6
    CHIRP_BW = 15e6
    SAMP_RATE = 16e6
    QUIT_AFTER_SECONDS = 5*60 #1200

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
