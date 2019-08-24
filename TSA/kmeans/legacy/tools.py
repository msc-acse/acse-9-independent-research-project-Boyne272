# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:20:45 2019

@author: Richard Bonye
"""

import sys

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
        m = int(length * i/i_max) + 1
        n = length - m
        sys.stdout.write("\rProgress |" + "#"*m + " "*n + "|")

        # update the string on screen
        sys.stdout.flush()
        return True

    elif i==i_max:
        sys.stdout.write("\rProgress |" + "#"*length + "|")
        sys.stdout.flush()
        return True

    return False
	
	
import numpy as np
from PIL import Image

def get_img(path):
    "Load an image as a numpy array"
    img = Image.open(path)
    arr = np.array(img.convert('RGB')).astype(float)/255.
    return arr
	

import torch
import random

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True
	
	
import time as tm
import matplotlib.pyplot as plt
import sys

class progress_bar():
    
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
        m = int(self.length * i/self.imax) + 1
        n = self.length - m
        sys.stdout.write("\rProgress |" + "#"*m + " "*n + "| %.4f s" % self.times[-1])
        
    def __call__(self, i):
        "if on correct iteration update the progress bar and store the time"
        if (i % self.refresh) == 0:
            self.add_time(i)
            self.print_bar(i)
            return True
        else:
            return False
        
    def plot_time(self, ax=None):
        "plot the time vs iterations on the axis if given"
        
        if not ax:
            fig, ax = plt.subplots()
            
        ax.plot(self.iterations, self.times, '-o')
        ax.set(xlabel="Iteration", ylabel="Time (s)")
        
        return ax