# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Wed Aug 28 08:46:31 2019
"""


import sys
import time as tm
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def percent_print(i, i_max, interval=1, length=50):
    """
    Print a progress bar or percentage value
    Parameters
    ----------
    i : int
        The current iteration number
    i_max : int
        The total number of iterations to complete
    interval : int, optional
        The frequency of updating the progress bar (set high for long
        simulations)
    Returns
    -------
    updated : bool
        Whether the update bar was updated
    """

    if i % interval == 0:

        # print percent
        # sys.stdout.write("\r %.1f %% Done" % (100*i/i_max))

        # print progress bar
        _m = int(length * i/i_max) + 1
        _n = length - _m
        sys.stdout.write("\rProgress |" + "#"*_m + " "*_n + "|")

        # update the string on screen
        sys.stdout.flush()
        return True

    if i == i_max:
        sys.stdout.write("\rProgress |" + "#" * length + "|")
        sys.stdout.flush()
        return True

    return False


def get_img(path):
    "Load an image as a numpy array"
    img = Image.open(path)
    arr = np.array(img.convert('RGB')).astype(float)/255.
    return arr


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


class progress_bar():
    """
    A simple progress bar for quick and easy user output
    """

    def __init__(self, imax, refresh=1, length=50):

        # store parameters
        self.imax = max(imax - 1, 1) # ensure a value of 1 wil not break it
        self.length = length
        self.refresh = refresh

        # create the time and iteration stores
        self.times = []
        self.iterations = []
        self.start = tm.clock()


    def add_time(self, iteration):
        "store the current time and iteration"
        self.times.append(tm.clock()-self.start)
        self.iterations.append(iteration)

    def print_bar(self, i):
        "update the progress bar"
        _m = int(self.length * i/self.imax) + 1
        _n = self.length - _m
        sys.stdout.write("\rProgress |" + "#" * _m + " " * _n +
                         "| %.4f s" % self.times[-1])

    def __call__(self, i):
        "if on correct iteration update the progress bar and store the time"
        if (i % self.refresh) == 0:
            self.add_time(i)
            self.print_bar(i)
            return True
        return False

    def plot_time(self, axis=None):
        "plot the time vs iterations on the axis if given"

        if axis is None:
            _fig, axis = plt.subplots()

        axis.plot(self.iterations, self.times, '-o')
        axis.set(xlabel="Iteration", ylabel="Time (s)")

        return axis
