import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import numpy as np
import time
import json
from typing import List, Tuple

from demo_types import DataSource

style.use("seaborn-v0_8")


class Anim:
    def __init__(
        self, fig: plt.Figure, data_source: DataSource, fft: bool = False
    ) -> None:
        self.start_time: float = time.time()
        self.end_time_seconds: float = (
            20  # how long to run the animation for, in seconds
        )
        self.data_source: DataSource = data_source
        self.data_dir: str = (
            "/Users/hannah/Documents/Novavon/uhd/host/novavon/sample_data/"
        )

        if self.data_source == DataSource.STATIC_JSON:
            self.animation: FuncAnimation = FuncAnimation(
                fig,
                self.animate,
                interval=1000,
                frames=5,
                repeat=True,
            )
        elif self.data_source == DataSource.CSV:
            self.xframes, self.yframes = self.load_csv_data("test_writer.csv")
            if fft:
                self.animation = FuncAnimation(
                    fig,
                    self.animate_fft,
                    interval=1000,
                    frames=20,
                    repeat=False,
                )
            else:
                self.x_data: List[float] = []
                self.y_data: List[float] = []
                self.animation = FuncAnimation(
                    fig,
                    self.animate_scrolling,
                    interval=2000,
                    frames=40,
                    repeat=False,
                )
        else:
            raise (NotImplementedError)

    def load_json_data(self, filename: str, imag: bool = False) -> Tuple[List, List]:
        x_data: List = []
        y1_data: List = []
        # y2_data = []

        with open(self.data_dir + filename, "r") as f:
            json_data = json.load(f)

        num_packets: int = len(json_data["packet_start"])
        delta_t: float = 1 / json_data.get("sampling_rate", 20e6)
        error_codes: List[str] = json_data.get("error_codes", ["NONE"] * num_packets)
        for ii in range(20, num_packets):
            if error_codes[ii] == "NONE":
                start: int = json_data["packet_start"][ii]
                num_samples: int = json_data["packet_size"][ii]
                xx: np.ndarray = np.linspace(
                    start, start + (num_samples - 1) * delta_t, num=num_samples
                )
                ch1: List[float] = json_data["packet_data_re"][ii][0]
                # ch2: List[float] = json_data["packet_data_re"][ii][1]
                if imag:
                    ch1 = [
                        complex(ch1[val], json_data["packet_data_im"][ii][0][val])
                        for val in range(len(ch1))
                    ]
                x_data = x_data + list(xx)
                y1_data = y1_data + list(ch1)
                # y2_data = y2_data + list(ch2)
        return x_data, y1_data

    def load_csv_data(self, filename: str) -> Tuple[List, List]:
        import csv
        import re

        x_data, y_data = [], []
        reg_chars = "[\[\\n\]]"
        with open(self.data_dir + filename, "r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                start = float(row["start_time"])
                fs = float(row["fs"])
                y_real = re.sub(reg_chars, "", row["real"]).split(" ")
                y_real = [float(val) for val in y_real if len(val)]

                y_imag = re.sub(reg_chars, "", row["imag"]).split(" ")
                y_imag = [float(val) for val in y_imag if len(val)]
                y = [complex(y_real[val], y_imag[val]) for val in range(len(y_real))]
                num_samples = len(y)
                x = np.linspace(
                    start, start + (num_samples - 1) / fs, num=num_samples
                ).tolist()
                x_data = x_data + x
                y_data = y_data + y
        return x_data, y_data

    def get_next_frame(self, frame_idx: int, frame_len: int) -> Tuple[List, List]:
        # for now the frames will come from self.xdat and self.ydat, but in future will come directly from sdr
        start_idx: int = frame_idx * frame_len
        stop_idx: int = start_idx + frame_len
        return self.xframes[start_idx:stop_idx], self.yframes[start_idx:stop_idx]

    def animate(self, i: int) -> None:
        if (i > 5) or ((time.time() - self.start_time) >= self.end_time_seconds):
            self.animation.event_source.stop()
        else:
            filename: str = f"sample_benchmark_data{i+1}.json"
            x_data, y_data = self.load_json_data(filename)
            title_str: str = filename

            plt.cla()
            plt.plot(x_data, y_data, label="Ch 1")
            # plt.plot(x_data, y2_data, label="Ch 2")
            plt.title(title_str)
            plt.xlabel("Time [milliseconds]")
            plt.ylabel("Real part")
            plt.legend(loc="upper right")
            # plt.xlim([1, 1.03])

    def animate_fft(self, i: int) -> None:
        # @TODO: shift to correct center freq

        frame_len: int = 500
        frame_t, frame_y = self.get_next_frame(i, frame_len)
        frame_Y = np.fft.fftshift(np.fft.fft(frame_y))
        delta_t = frame_t[1] - frame_t[0]
        frame_f = np.fft.fftshift(np.fft.fftfreq(frame_len, delta_t))

        plt.cla()
        plt.plot(frame_f, 20 * np.log10(abs(frame_Y)))
        plt.title(f"Frame {i+1}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.ylim([-30, 30])

    def animate_scrolling(self, i: int) -> None:
        frame_len: int = 500
        frames_to_show: int = 2
        shift: int = frame_len  # int(0.5 * frame_len)

        frame_x, frame_y = self.get_next_frame(i, frame_len)

        self.x_data = self.x_data + frame_x
        self.y_data = self.y_data + np.real(frame_y).tolist()

        if len(self.y_data) > frame_len:
            if i >= frames_to_show:
                self.x_data = self.x_data[shift:]
                self.y_data = self.y_data[shift:]

            if max(self.y_data) > 0.1:
                plt.cla()
                plt.plot(self.x_data, self.y_data, label="Ch 1")
                plt.title(f"Frame {i+1}")
                plt.xlabel("Time [milliseconds]")
                plt.ylabel("Real part")


def main():
    data_source: DataSource = DataSource.CSV
    fft = False
    fig: plt.Figure = plt.figure()
    a: Anim = Anim(fig, data_source, fft=fft)
    plt.show()


if __name__ == "__main__":
    main()
