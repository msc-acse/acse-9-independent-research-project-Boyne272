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