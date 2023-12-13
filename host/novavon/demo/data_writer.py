import csv
import numpy as np
import time


def main():
    output_path = "/Users/hannah/Documents/Novavon/uhd/host/novavon/sample_data/"
    filename = "test_writer.csv"
    fs = 20e6
    packet_len = 500
    packet_start = 0.95 + np.random.rand(1) / 10
    packet_start = packet_start[0]
    t = np.linspace(0, fs * packet_len, packet_len)

    fieldnames = ["start_time", "fs", "real", "imag"]

    start_time = time.time()
    current_time = start_time

    with open(output_path + filename, "w") as f:
        # start with a blank file
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

    while (current_time - start_time) < 100:
        data_real = 0.1 * np.random.rand(packet_len)
        data_real[20:] = data_real[20:] + np.cos(t[20:])

        data_imag = 0.1 * np.random.rand(packet_len)
        data_imag[20:] = data_imag[20:] + np.sin(t[20:])

        data_out = {
            "start_time": packet_start,
            "fs": fs,
            "real": data_real,
            "imag": data_imag,
        }

        with open(output_path + filename, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writerow(data_out)

        packet_start = packet_start + 0.00005
        time.sleep(1)
        current_time = time.time()


if __name__ == "__main__":
    main()
