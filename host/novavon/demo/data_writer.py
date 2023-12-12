import csv
import numpy as np
import time


def main():
    output_path = "/Users/hannah/Documents/Novavon/uhd/host/novavon/sample_data/"
    filename = "test_writer.csv"
    fs = 20e6
    packet_len = 500
    packet_start = 0.95 + np.random.rand(1) / 10
    t = np.linspace(0, fs * packet_len, packet_len)

    fieldnames = ["start_time", "fs", "real", "imag"]

    with open(output_path + filename, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

    while True:
        data_real = 0.1 * np.random.rand(packet_len)
        data_real[200:] = data_real[200:] + np.cos(t[200:])

        data_imag = 0.1 * np.random.rand(packet_len)
        data_imag[200:] = data_imag[200:] + np.sin(t[200:])

        data_out = {
            "start_time": packet_start[0],
            "fs": fs,
            "real": data_real,
            "imag": data_imag,
        }

        with open(output_path + filename, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writerow(data_out)

        packet_start = packet_start + 0.0002
        time.sleep(1.5)


if __name__ == "__main__":
    main()
