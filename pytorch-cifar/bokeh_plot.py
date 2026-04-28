import os
from pathlib import Path
import datetime


def parse_log_entries(log_path):
    lines, dttm = [], None
    with open(log_path, "rt") as f:
        for line in f.readlines():
            try:
                next_dttm = datetime.datetime.strptime(line[0:19], "%Y-%m-%d %H:%M:%S")
                next_dttm = next_dttm + datetime.timedelta(
                    milliseconds=int(line[20:23])
                )

                if dttm:
                    yield (dttm, entry_type, lines)

                # now add the current data
                dttm = next_dttm
                remainder = line[24:].split(" ")
                entry_type, lines = remainder[0], [" ".join(remainder[1:]).strip("\n")]

            except ValueError:
                lines.append(line.strip("\n"))


def scan_log_entries():
    for log_path in sorted(Path("checkpoint").glob("*/log.txt")):
        yield (log_path.parts[-2], parse_log_entries(log_path))


def test_scan_entries():
    for (folder, entries) in scan_log_entries():
        print(folder)
        for (dttm, entry_type, lines) in entries:
            if len(lines) == 1:
                print(f"\t{dttm} {entry_type} {lines[0]}")
            else:
                print(f"\t{dttm} {entry_type}")
                for line in lines:
                    print(f"\t\t{line}")


def iteration_entry_to_dict(entry):
    parts = list(entry.split("|"))
    entry_dict = {}
    entry_dict["epoch"] = parts[1].split(":")[1].trim(" ")
    entry_dict["batch"] = parts[2].split(":")[1].trim(" ")
    entry_dict["loss"] = parts[3].split(":")[1].trim(" ")
    entry_dict["accuracy"] = parts[4].split(":")[1].trim(" ").split(" ")[0]
    print(entry_dict)
    return entry_dict


def extract_training():
    for (folder, entries) in scan_log_entries():
        train_iterations = [
            iteration_entry_to_dict(entry_lines[0])
            for (_,_,entry_lines) in entries
            if len(entry_lines) == 1 and entry_lines[0].startswith("train iteration")
        ]        

extract_training()

import numpy as np

import torch
from torchvision.datasets import CIFAR10

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.themes import Theme
from bokeh.server.server import Server


def make_test_image(N=20):
    # create an array of RGBA data
    img = np.empty((N, N), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((N, N, 4))
    for i in range(N):
        for j in range(N):
            view[i, j, 0] = int(255 * i / N)
            view[i, j, 1] = 158
            view[i, j, 2] = int(255 * j / N)
            view[i, j, 3] = 255
    return img


num_classes = 10
cifar10_testset = CIFAR10(root="./data", train=False, download=True, transform=None)
# print(f"{cifar10_testset.data[1].shape}")


def rgb_to_uint32(x):
    # print(x.shape)
    rgba = bytearray(x.tobytes())
    rgba.append(255)
    return np.frombuffer(rgba, dtype=np.uint32)


def get_cifar_images(N=10):
    rgba_images = [
        np.apply_along_axis(rgb_to_uint32, -1, cifar10_testset.data[n])
        for n in range(N)
    ]
    rgba_images = map(np.flipud, rgba_images)
    rgba_images = map(np.squeeze, rgba_images)
    rgba_images = list(rgba_images)
    # print(rgba_images)
    return rgba_images


def bkapp(doc):
    df = sea_surface_temperature.copy()
    source = ColumnDataSource(data=df)

    plot = figure(width=400, height=400, x_range=(-10, 10), y_range=(-10, 10))

    # plot.line("time", "temperature", source=source)

    test_img = make_test_image()
    # plot.image_rgba(image=[test_img], x=[2], y=[2], dw=[3], dh=[3])

    N = 100
    cifar_imgs = get_cifar_images(N)
    plot.image_rgba(
        image=cifar_imgs,
        x=np.random.randint(low=-10, high=10, size=(N,)),
        y=np.random.randint(low=-10, high=10, size=(N,)),
        dw=1,
        dh=1,
    )

    def callback(attr, old, new):
        if new == 0:
            data = df
        else:
            data = df.rolling(f"{new}D").mean()
        source.data = ColumnDataSource.from_df(data)

    slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    slider.on_change("value", callback)

    doc.add_root(column(slider, plot))
    # doc.add_root(p)

    doc.theme = Theme(filename="theme.yaml")


# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({"/": bkapp}, num_procs=1)
server.start()

if __name__ == "__main__":
    print("Opening Bokeh application on http://localhost:5006/")

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
