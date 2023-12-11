import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import style
import numpy as np
import time
import json

style.use("seaborn-v0_8")


class Anim:
    def __init__(self, fig, **kw):
        self.start_time = time.time()
        self.end_time_seconds = 20  # how long to run the animation for, in seconds
        self.animation = FuncAnimation(
            fig, self.animate, interval=1000, frames=5, repeat=True
        )
        self.data_dir = "/Users/hannah/Documents/Novavon/uhd/host/novavon/sample_data/"

    def animate(self, i):
        if (i > 5) or ((time.time() - self.start_time) >= self.end_time_seconds):
            self.animation.event_source.stop()
        else:
            filename = f"sample_benchmark_data{i+1}.json"
            x_data = []
            y1_data = []
            y2_data = []

            with open(self.data_dir + filename, "r") as f:
                json_data = json.load(f)

            num_packets = len(json_data["packet_start"])
            delta_t = 1 / json_data.get("sampling_rate", 20e6)
            error_codes = json_data.get("error_codes", ["NONE"] * num_packets)
            for ii in range(30, num_packets):
                if error_codes[ii] == "NONE":
                    start = json_data["packet_start"][ii]
                    num_samples = json_data["packet_size"][ii]
                    xx = np.linspace(
                        start, start + (num_samples - 1) * delta_t, num=num_samples
                    )
                    ch1 = json_data["packet_data_re"][ii][0]
                    ch2 = json_data["packet_data_re"][ii][1]
                    x_data = x_data + list(xx)
                    y1_data = y1_data + list(ch1)
                    y2_data = y2_data + list(ch2)

            # plot
            plt.cla()
            plt.plot(x_data, y1_data, label="Ch 1")
            plt.plot(x_data, y2_data, label="Ch 2")
            plt.title(filename)
            plt.xlabel("Time [milliseconds]")
            plt.ylabel("Real part")
            plt.legend(loc="upper right")
            # plt.xlim([1, 1.03])


def main():
    fig = plt.figure()
    a = Anim(fig)
    plt.show()


if __name__ == "__main__":
    main()
